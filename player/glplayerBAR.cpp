#define _CRT_SECURE_NO_WARNINGS
#include <QApplication>
#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <QOpenGLShaderProgram>
#include <QDebug>
#include <QTimer>
#include <QElapsedTimer>
#include <QPainter>
#include <QMouseEvent>
#include <functional>

#include <thread>
#include <mutex>
#include <atomic>
#include <cstdint>
#include <condition_variable>
#include <deque>

#include <QAudioOutput>
#include <QAudioFormat>
#include <QAudioDeviceInfo>
#include <QIODevice>
#include <vector>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
#include <libavutil/pixdesc.h>
#include <libswresample/swresample.h>
#include <libavutil/opt.h>
#include <libavutil/channel_layout.h>
}

#ifndef GL_UNPACK_ROW_LENGTH
#  define GL_UNPACK_ROW_LENGTH 0x0CF2
#endif
#ifndef GL_LUMINANCE
#  define GL_LUMINANCE 0x1909
#endif

// ==================== 着色器 ====================
static const char* kVS = R"(#version 120
attribute vec2 aPos;
attribute vec2 aTex;
varying vec2 vTex;
void main()
{
    vTex = vec2(aTex.x, 1.0 - aTex.y);
    gl_Position = vec4(aPos, 0.0, 1.0);
}
)";

static const char* kFS = R"(#version 120
varying vec2 vTex;
uniform sampler2D uY;
uniform sampler2D uU;
uniform sampler2D uV;
void main()
{
    float y = texture2D(uY, vTex).r;
    float u = texture2D(uU, vTex).r - 0.5;
    float v = texture2D(uV, vTex).r - 0.5;

    y = (y - 16.0 / 255.0) * 1.164;

    vec3 rgb = vec3(y + 1.596 * v,
                    y - 0.813 * v - 0.391 * u,
                    y + 2.018 * u);

    gl_FragColor = vec4(clamp(rgb, 0.0, 1.0), 1.0);
}
)";

// ==================== ① 帧队列（带世代号） ====================
struct QueuedFrame
{
    AVFrame* frame = nullptr;
    double   dueMs = 0.0;
};

class FrameQueue
{
public:
    explicit FrameQueue(size_t maxSize = 6) : m_max(maxSize) {}

    ~FrameQueue()
    {
        stop();
        std::lock_guard<std::mutex> lk(m_mtx);
        for (auto& qf : m_q) av_frame_free(&qf.frame);
        m_q.clear();
    }

    // 当前世代号：解码线程在「开始解一帧之前」读它，push 时带回来
    uint64_t epoch() const { return m_epoch.load(); }

    // 解码线程：放货。队列满了就睡着等（背压）
    // ep = 解这一帧时的世代号；和当前世代不一致说明中途发生过 seek，扔掉
    bool push(AVFrame* f, double dueMs, uint64_t ep)
    {
        std::unique_lock<std::mutex> lk(m_mtx);
        m_cv.wait(lk, [this] { return m_q.size() < m_max || m_stop; });
        if (m_stop) { av_frame_free(&f); return false; }

        // ★ 世代过期：这是 seek 之前开始解的旧帧，扔掉
        if (ep != m_epoch.load()) { av_frame_free(&f); return true; }

        // 兜底：低于 floor 的也不收
        if (dueMs < m_floorMs) { av_frame_free(&f); return true; }

        m_q.push_back({ f, dueMs });
        m_cv.notify_all();
        return true;
    }

    // 显示端：把到点的都取走，只留最新的，更旧的释放（这就是丢帧）
    AVFrame* takeLatestDue(double nowMs, double* outDueMs = nullptr)
    {
        std::unique_lock<std::mutex> lk(m_mtx);
        if (m_q.empty() || m_q.front().dueMs > nowMs) return nullptr;

        AVFrame* latest = nullptr;
        double   latestDue = 0.0;
        while (!m_q.empty() && m_q.front().dueMs <= nowMs)
        {
            if (latest) { av_frame_free(&latest); ++m_dropped; }
            latest = m_q.front().frame;
            latestDue = m_q.front().dueMs;
            m_q.pop_front();
        }
        if (outDueMs) *outDueMs = latestDue;
        m_cv.notify_all();
        return latest;
    }

    // ★ seek：清空队列 + 世代 +1 + 记下限
    void clear(double fromMs = 0.0)
    {
        std::lock_guard<std::mutex> lk(m_mtx);
        for (auto& qf : m_q) av_frame_free(&qf.frame);
        m_q.clear();
        m_floorMs = fromMs;
        ++m_epoch;                       // 世代前进，之前的帧全部作废
        m_cv.notify_all();
    }

    bool empty() { std::lock_guard<std::mutex> lk(m_mtx); return m_q.empty(); }
    long dropped() { std::lock_guard<std::mutex> lk(m_mtx); return m_dropped; }

    void markFinished() { std::lock_guard<std::mutex> lk(m_mtx); m_finished = true; m_cv.notify_all(); }
    bool finished() { std::lock_guard<std::mutex> lk(m_mtx); return m_finished; }

    void stop()
    {
        { std::lock_guard<std::mutex> lk(m_mtx); m_stop = true; }
        m_cv.notify_all();
    }

private:
    std::mutex              m_mtx;
    std::condition_variable m_cv;
    std::deque<QueuedFrame> m_q;
    size_t                  m_max;
    double                  m_floorMs = 0.0;
    std::atomic<uint64_t>   m_epoch{ 0 };     // ★ 世代号
    bool                    m_stop = false;
    bool                    m_finished = false;
    long                    m_dropped = 0;
};

// ==================== ② 视频解码器 ====================
class Decoder
{
public:
    void requestSeek(double ms) { m_seekMs.store((int64_t)ms); }

    bool open(const char* path)
    {
        if (avformat_open_input(&m_fmt, path, nullptr, nullptr) < 0) return false;
        if (avformat_find_stream_info(m_fmt, nullptr) < 0) return false;

        m_idx = av_find_best_stream(m_fmt, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
        if (m_idx < 0) return false;

        m_tb = m_fmt->streams[m_idx]->time_base;

        const AVCodecParameters* par = m_fmt->streams[m_idx]->codecpar;
        const AVCodec* dec = avcodec_find_decoder(par->codec_id);
        if (!dec) return false;
        m_ctx = avcodec_alloc_context3(dec);
        if (!m_ctx) return false;
        avcodec_parameters_to_context(m_ctx, par);
        if (avcodec_open2(m_ctx, dec, nullptr) < 0) return false;

        m_pkt = av_packet_alloc();
        m_tmp = av_frame_alloc();
        return m_pkt && m_tmp;
    }

    AVFrame* nextOwnedFrame()
    {
        const int64_t req = m_seekMs.exchange(-1);
        if (req >= 0) doSeek(req);

        for (int guard = 0; guard < 20000; ++guard)
        {
            int ret = avcodec_receive_frame(m_ctx, m_tmp);
            if (ret == 0)
            {
                if (m_firstPts == AV_NOPTS_VALUE && m_tmp->pts != AV_NOPTS_VALUE)
                    m_firstPts = m_tmp->pts;

                if (m_tmp->pts != AV_NOPTS_VALUE && m_tmp->pts < m_dropBeforePts) {
                    ++m_dropCount;
                    continue;
                }
                if (m_dropCount > 0) {
                    qInfo() << "[视频] 空转结束，丢掉" << m_dropCount << "帧后追上目标";
                    m_dropCount = 0;
                }
                m_dropBeforePts = INT64_MIN;

                AVFrame* out = av_frame_alloc();
                if (!out) return nullptr;
                av_frame_move_ref(out, m_tmp);
                return out;
            }
            if (ret == AVERROR_EOF) {
                qWarning() << "[视频] receive_frame 返回 EOF → 线程退出";
                return nullptr;
            }

            if (av_read_frame(m_fmt, m_pkt) < 0)
            {
                if (m_flushed) {
                    qWarning() << "[视频] 读流二次失败 → 线程退出";
                    return nullptr;
                }
                m_flushed = true;
                avcodec_send_packet(m_ctx, nullptr);
                continue;
            }
            if (m_pkt->stream_index != m_idx) { av_packet_unref(m_pkt); continue; }

            ret = avcodec_send_packet(m_ctx, m_pkt);
            av_packet_unref(m_pkt);
            if (ret < 0 && ret != AVERROR(EAGAIN)) continue;
        }
        qWarning() << "[视频] 守卫耗尽 → 线程退出";
        return nullptr;
    }

    double dueMs(int64_t pts) const
    {
        if (pts == AV_NOPTS_VALUE || m_firstPts == AV_NOPTS_VALUE) return 0.0;
        return (pts - m_firstPts) * av_q2d(m_tb) * 1000.0;
    }

    double durationMs() const
    {
        if (!m_fmt || m_fmt->duration == AV_NOPTS_VALUE) return 0.0;
        return m_fmt->duration / 1000.0;
    }

    void close()
    {
        if (m_tmp) av_frame_free(&m_tmp);
        if (m_pkt) av_packet_free(&m_pkt);
        if (m_ctx) avcodec_free_context(&m_ctx);
        if (m_fmt) avformat_close_input(&m_fmt);
    }

private:
    void doSeek(int64_t ms)
    {
        if (!m_fmt || m_firstPts == AV_NOPTS_VALUE) return;

        const double tbMs = av_q2d(m_tb) * 1000.0;
        const int64_t targetPts = m_firstPts + (int64_t)((double)ms / tbMs);

        if (av_seek_frame(m_fmt, m_idx, targetPts, AVSEEK_FLAG_BACKWARD) < 0)
            qWarning() << "[视频] av_seek_frame 失败";

        avcodec_flush_buffers(m_ctx);
        m_dropBeforePts = targetPts;
        m_dropCount = 0;
        m_flushed = false;
        qInfo() << "[视频] seek 完成，空转到" << ms << "ms";
    }

    AVFormatContext* m_fmt = nullptr;
    AVCodecContext* m_ctx = nullptr;
    AVFrame* m_tmp = nullptr;
    AVPacket* m_pkt = nullptr;
    int              m_idx = -1;
    AVRational       m_tb = { 1, 1000 };
    int64_t          m_firstPts = AV_NOPTS_VALUE;
    int64_t          m_dropBeforePts = INT64_MIN;
    int              m_dropCount = 0;
    std::atomic<int64_t> m_seekMs{ -1 };
    bool             m_flushed = false;
};

// ==================== ③ View ====================
class View : public QOpenGLWidget, protected QOpenGLFunctions
{
public:
    std::function<void(double)> onSeek;

    void setFrame(const uint8_t* y, int lsY,
        const uint8_t* u, int lsU,
        const uint8_t* v, int lsV,
        int w, int h)
    {
        m_plane[0] = y;  m_linesize[0] = lsY;
        m_plane[1] = u;  m_linesize[1] = lsU;
        m_plane[2] = v;  m_linesize[2] = lsV;
        m_w = w;  m_h = h;
        repaint();
    }

    void setDuration(double ms) { m_durationMs = ms; }
    void setPosition(double ms) { m_positionMs = ms; }

    static QString fmtTime(double ms)
    {
        if (ms < 0.0) return QString("--:--");
        const int total = (int)(ms / 1000.0);
        return QString("%1:%2")
            .arg(total / 60, 2, 10, QChar('0'))
            .arg(total % 60, 2, 10, QChar('0'));
    }

protected:
    void initializeGL() override
    {
        initializeOpenGLFunctions();
        glClearColor(0.1f, 0.1f, 0.12f, 1.0f);
        m_prog = new QOpenGLShaderProgram(this);
        m_prog->addShaderFromSourceCode(QOpenGLShader::Vertex, kVS);
        m_prog->addShaderFromSourceCode(QOpenGLShader::Fragment, kFS);
        m_prog->bindAttributeLocation("aPos", 0);
        m_prog->bindAttributeLocation("aTex", 1);
        if (!m_prog->link())
            qWarning() << "shader 失败:" << m_prog->log();
        glGenTextures(3, m_tex);
        for (int i = 0; i < 3; ++i) {
            glBindTexture(GL_TEXTURE_2D, m_tex[i]);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        }
    }

    void paintGL() override
    {
        glClear(GL_COLOR_BUFFER_BIT);
        const float verts[] = {
            -1.0f, -1.0f,  0.0f, 0.0f,
             1.0f, -1.0f,  1.0f, 0.0f,
            -1.0f,  1.0f,  0.0f, 1.0f,
             1.0f,  1.0f,  1.0f, 1.0f,
        };
        if (m_plane[0])
        {
            const int cw = (m_w + 1) / 2;
            const int ch = (m_h + 1) / 2;
            uploadPlane(0, m_tex[0], m_w, m_h);
            uploadPlane(1, m_tex[1], cw, ch);
            uploadPlane(2, m_tex[2], cw, ch);
        }
        m_prog->bind();
        m_prog->setUniformValue("uY", 0);
        m_prog->setUniformValue("uU", 1);
        m_prog->setUniformValue("uV", 2);
        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, m_tex[0]);
        glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, m_tex[1]);
        glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_2D, m_tex[2]);
        glEnableVertexAttribArray(0);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), verts);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), verts + 2);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        glDisableVertexAttribArray(0);
        glDisableVertexAttribArray(1);
        m_prog->release();

        for (int i = 0; i < 3; ++i) {
            glActiveTexture(GL_TEXTURE0 + i);
            glBindTexture(GL_TEXTURE_2D, 0);
        }
        glActiveTexture(GL_TEXTURE0);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
        glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
        glDisable(GL_BLEND);

        QPainter p(this);
        p.setRenderHint(QPainter::Antialiasing, true);
        drawProgressBar(p);
        p.end();
    }

    void mousePressEvent(QMouseEvent* e) override
    {
        if (e->button() != Qt::LeftButton) return;
        m_dragging = true;
        trySeekAt(e->pos(), true);
    }

    void mouseMoveEvent(QMouseEvent* e) override
    {
        if (!m_dragging) return;
        trySeekAt(e->pos(), false);
    }

    void mouseReleaseEvent(QMouseEvent* e) override
    {
        if (e->button() != Qt::LeftButton || !m_dragging) return;
        m_dragging = false;
        trySeekAt(e->pos(), true);
    }

private:
    QRect barRect() const
    {
        return QRect(24, height() - 30, width() - 48, 6);
    }

    void trySeekAt(const QPoint& pos, bool commit)
    {
        if (m_durationMs <= 0.0) return;

        const QRect bar = barRect();
        if (pos.y() < bar.top() - 50) return;

        double t = (double)(pos.x() - bar.left()) / (double)bar.width();
        t = qBound(0.0, t, 1.0);
        const double targetMs = t * m_durationMs;

        m_positionMs = targetMs;
        repaint();

        if (commit && onSeek) onSeek(targetMs);
    }

    void uploadPlane(int i, GLuint tex, int w, int h)
    {
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, w, h, 0,
            GL_LUMINANCE, GL_UNSIGNED_BYTE, nullptr);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glPixelStorei(GL_UNPACK_ROW_LENGTH, m_linesize[i]);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h,
            GL_LUMINANCE, GL_UNSIGNED_BYTE, m_plane[i]);
        glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
    }

    void drawProgressBar(QPainter& p)
    {
        const int margin = 24;
        const int barH = 6;
        const int barY = height() - 30;
        const QRect bar(margin, barY, width() - margin * 2, barH);
        if (bar.width() <= 0) return;

        p.setPen(Qt::NoPen);
        p.setBrush(QColor(255, 255, 255, 60));
        p.drawRoundedRect(bar, 3, 3);

        if (m_durationMs > 0.0)
        {
            const double t = qBound(0.0, m_positionMs / m_durationMs, 1.0);
            const int wDone = (int)(bar.width() * t);
            if (wDone > 0) {
                p.setBrush(QColor(70, 170, 255));
                p.drawRoundedRect(QRect(bar.left(), bar.top(), wDone, bar.height()), 3, 3);
            }
        }

        QFont f = p.font();
        f.setPointSize(9);
        p.setFont(f);
        p.setPen(QColor(255, 255, 255, 220));
        p.drawText(QRect(margin, barY - 22, 200, 18),
            Qt::AlignLeft | Qt::AlignVCenter, fmtTime(m_positionMs));
        p.drawText(QRect(width() - margin - 200, barY - 22, 200, 18),
            Qt::AlignRight | Qt::AlignVCenter,
            m_durationMs > 0.0 ? fmtTime(m_durationMs) : QString("--:--"));
    }

    QOpenGLShaderProgram* m_prog = nullptr;
    GLuint         m_tex[3] = { 0, 0, 0 };
    const uint8_t* m_plane[3] = { nullptr, nullptr, nullptr };
    int            m_linesize[3] = { 0, 0, 0 };
    int            m_w = 0, m_h = 0;

    double         m_durationMs = 0.0;
    double         m_positionMs = 0.0;
    bool           m_dragging = false;
};

// ==================== ④ PCM 队列（也带世代号） ====================
class PcmQueue
{
public:
    explicit PcmQueue(size_t maxBytes = 96000) : m_max(maxBytes) {}
    ~PcmQueue()
    {
        stop();
        std::lock_guard<std::mutex> lk(m_mtx);
        m_q.clear();
    }

    uint64_t epoch() const { return m_epoch.load(); }

    bool push(std::vector<uint8_t>&& chunk, uint64_t ep)
    {
        std::unique_lock<std::mutex> lk(m_mtx);
        m_cv.wait(lk, [this] { return m_bytes < m_max || m_stop; });
        if (m_stop) return false;
        if (ep != m_epoch.load()) return true;      // ★ 旧世代的音频块，扔

        m_bytes += chunk.size();
        m_q.push_back(std::move(chunk));
        m_cv.notify_all();
        return true;
    }

    bool popIfFits(int freeBytes, std::vector<uint8_t>& out)
    {
        std::lock_guard<std::mutex> lk(m_mtx);
        if (m_q.empty()) return false;
        if ((int)m_q.front().size() > freeBytes) return false;
        out = std::move(m_q.front());
        m_q.pop_front();
        m_bytes -= out.size();
        m_cv.notify_all();
        return true;
    }

    void clear()
    {
        std::lock_guard<std::mutex> lk(m_mtx);
        m_q.clear();
        m_bytes = 0;
        ++m_epoch;
        m_cv.notify_all();
    }

    bool empty() { std::lock_guard<std::mutex> lk(m_mtx); return m_q.empty(); }
    void markFinished() { std::lock_guard<std::mutex> lk(m_mtx); m_finished = true; }
    bool finished() { std::lock_guard<std::mutex> lk(m_mtx); return m_finished; }
    void stop()
    {
        { std::lock_guard<std::mutex> lk(m_mtx); m_stop = true; }
        m_cv.notify_all();
    }
private:
    std::mutex                         m_mtx;
    std::condition_variable            m_cv;
    std::deque<std::vector<uint8_t>>   m_q;
    size_t                             m_max;
    size_t                             m_bytes = 0;
    std::atomic<uint64_t>              m_epoch{ 0 };
    bool                               m_stop = false;
    bool                               m_finished = false;
};

// ==================== ⑤ 音频解码器 ====================
class AudioDecoder
{
public:
    void requestSeek(double ms) { m_seekMs.store((int64_t)ms); }

    bool open(const char* path, int outRate, int outCh)
    {
        if (avformat_open_input(&m_fmt, path, nullptr, nullptr) < 0) return false;
        if (avformat_find_stream_info(m_fmt, nullptr) < 0) return false;

        m_idx = av_find_best_stream(m_fmt, AVMEDIA_TYPE_AUDIO, -1, -1, nullptr, 0);
        if (m_idx < 0) return false;

        m_tb = m_fmt->streams[m_idx]->time_base;

        const AVCodecParameters* par = m_fmt->streams[m_idx]->codecpar;
        const AVCodec* dec = avcodec_find_decoder(par->codec_id);
        if (!dec) return false;
        m_ctx = avcodec_alloc_context3(dec);
        if (!m_ctx) return false;
        avcodec_parameters_to_context(m_ctx, par);
        if (avcodec_open2(m_ctx, dec, nullptr) < 0) return false;

        m_pkt = av_packet_alloc();
        m_frame = av_frame_alloc();

        m_outRate = outRate;
        m_outCh = outCh;
        av_channel_layout_default(&m_outLayout, outCh);
        return true;
    }

    bool nextChunk(std::vector<uint8_t>& out)
    {
        out.clear();

        const int64_t req = m_seekMs.exchange(-1);
        if (req >= 0) doSeek(req);

        for (int guard = 0; guard < 20000; ++guard)
        {
            int ret = avcodec_receive_frame(m_ctx, m_frame);
            if (ret == 0)
            {
                if (m_firstPts == AV_NOPTS_VALUE && m_frame->pts != AV_NOPTS_VALUE)
                    m_firstPts = m_frame->pts;

                if (m_frame->pts != AV_NOPTS_VALUE && m_frame->pts < m_dropBeforePts) {
                    av_frame_unref(m_frame);
                    continue;
                }
                m_dropBeforePts = INT64_MIN;

                convert(out);
                av_frame_unref(m_frame);
                return true;
            }
            if (ret == AVERROR_EOF) {
                qWarning() << "[音频] receive_frame 返回 EOF → 线程退出";
                return false;
            }

            if (av_read_frame(m_fmt, m_pkt) < 0)
            {
                if (m_flushed) {
                    qWarning() << "[音频] 读流二次失败 → 线程退出";
                    return false;
                }
                m_flushed = true;
                avcodec_send_packet(m_ctx, nullptr);
                continue;
            }
            if (m_pkt->stream_index != m_idx) { av_packet_unref(m_pkt); continue; }

            ret = avcodec_send_packet(m_ctx, m_pkt);
            av_packet_unref(m_pkt);
            if (ret < 0 && ret != AVERROR(EAGAIN)) continue;
        }
        qWarning() << "[音频] 守卫耗尽 → 线程退出";
        return false;
    }

    void close()
    {
        if (m_swr)   swr_free(&m_swr);
        if (m_frame) av_frame_free(&m_frame);
        if (m_pkt)   av_packet_free(&m_pkt);
        if (m_ctx)   avcodec_free_context(&m_ctx);
        if (m_fmt)   avformat_close_input(&m_fmt);
    }
    ~AudioDecoder() { close(); }

private:
    void doSeek(int64_t ms)
    {
        if (!m_fmt) return;

        const double tbMs = av_q2d(m_tb) * 1000.0;
        const int64_t base = (m_firstPts == AV_NOPTS_VALUE) ? 0 : m_firstPts;
        const int64_t targetPts = base + (int64_t)((double)ms / tbMs);

        if (av_seek_frame(m_fmt, m_idx, targetPts, AVSEEK_FLAG_BACKWARD) < 0)
            qWarning() << "[音频] av_seek_frame 失败";

        avcodec_flush_buffers(m_ctx);
        if (m_swr) swr_free(&m_swr);

        m_dropBeforePts = targetPts;
        m_flushed = false;
        qInfo() << "[音频] seek 完成";
    }

    void convert(std::vector<uint8_t>& out)
    {
        if (!m_swr)
        {
            if (swr_alloc_set_opts2(&m_swr,
                &m_outLayout, AV_SAMPLE_FMT_S16, m_outRate,
                &m_frame->ch_layout, (AVSampleFormat)m_frame->format,
                m_frame->sample_rate, 0, nullptr) < 0) return;
            if (swr_init(m_swr) < 0) return;
            qInfo() << "[音频] 转换:" << m_frame->sample_rate << "Hz /"
                << m_frame->ch_layout.nb_channels << "ch /"
                << av_get_sample_fmt_name((AVSampleFormat)m_frame->format)
                << "→" << m_outRate << "Hz /" << m_outCh << "ch / s16";
        }

        const int maxOut = (int)av_rescale_rnd(
            swr_get_delay(m_swr, m_frame->sample_rate) + m_frame->nb_samples,
            m_outRate, m_frame->sample_rate, AV_ROUND_UP);

        out.resize((size_t)maxOut * m_outCh * 2);
        uint8_t* p = out.data();
        const int got = swr_convert(m_swr, &p, maxOut,
            (const uint8_t**)m_frame->extended_data,
            m_frame->nb_samples);
        out.resize(got > 0 ? (size_t)got * m_outCh * 2 : 0);
    }

    AVFormatContext* m_fmt = nullptr;
    AVCodecContext* m_ctx = nullptr;
    AVFrame* m_frame = nullptr;
    AVPacket* m_pkt = nullptr;
    SwrContext* m_swr = nullptr;
    int              m_idx = -1;
    int              m_outRate = 48000;
    int              m_outCh = 2;
    AVRational       m_tb = { 1, 1000 };
    int64_t          m_firstPts = AV_NOPTS_VALUE;
    int64_t          m_dropBeforePts = INT64_MIN;
    std::atomic<int64_t> m_seekMs{ -1 };
    AVChannelLayout  m_outLayout{};
    bool             m_flushed = false;
};


int main(int argc, char* argv[])
{
    QApplication app(argc, argv);

    // ========== ① 音频输出格式协商 ==========
    QAudioFormat fmt;
    fmt.setSampleRate(48000);
    fmt.setChannelCount(2);
    fmt.setSampleSize(16);
    fmt.setCodec("audio/pcm");
    fmt.setByteOrder(QAudioFormat::LittleEndian);
    fmt.setSampleType(QAudioFormat::SignedInt);

    const QAudioDeviceInfo info = QAudioDeviceInfo::defaultOutputDevice();
    if (!info.isFormatSupported(fmt)) fmt = info.nearestFormat(fmt);
    qInfo() << "声卡:" << info.deviceName()
        << fmt.sampleRate() << "Hz" << fmt.channelCount() << "ch";

    const int outRate = fmt.sampleRate();
    const int outCh = fmt.channelCount();

    QAudioOutput audio(fmt);
    QIODevice* audioDev = audio.start();
    if (!audioDev) { qCritical() << "打开声卡失败"; return 1; }

    // ========== ② 队列与解码器 ==========
    FrameQueue   vq(6);
    PcmQueue     aq(96000);
    Decoder      vdec;
    AudioDecoder adec;
    QElapsedTimer clock;

    const char* path = "C:/Users/86150/Videos/2026-07-02 16-02-22.mkv";

    // ========== ③ 窗口 ==========
    View view;
    view.resize(1280, 720);
    view.setWindowTitle("⑦-5：进度条（双向跳转）");
    view.show();

    clock.start();

    std::atomic<double> totalMs{ 0.0 };

    // ========== ④ 视频解码线程 ==========
    std::thread vthread([&]() {
        if (!vdec.open(path)) { qWarning() << "视频打不开"; vq.markFinished(); return; }
        totalMs.store(vdec.durationMs());
        for (;;) {
            const uint64_t ep = vq.epoch();          // ★ 解之前取世代号
            AVFrame* f = vdec.nextOwnedFrame();
            if (!f) break;
            if (!vq.push(f, vdec.dueMs(f->pts), ep)) break;
        }
        qInfo() << "[视频] 解码线程结束";
        vq.markFinished();
        });

    // ========== ⑤ 音频解码线程 ==========
    std::thread athread([&]() {
        if (!adec.open(path, outRate, outCh)) { qWarning() << "音频打不开"; aq.markFinished(); return; }
        for (;;) {
            const uint64_t ep = aq.epoch();          // ★ 同上
            std::vector<uint8_t> chunk;
            if (!adec.nextChunk(chunk)) break;
            if (chunk.empty()) continue;
            if (!aq.push(std::move(chunk), ep)) break;
        }
        qInfo() << "[音频] 解码线程结束";
        aq.markFinished();
        });

    // ========== ⑥ 界面定时器 ==========
    uint64_t audioWritten = 0;
    double   audioBaseMs = 0.0;
    const int bytesPerSec = outRate * outCh * 2;

    auto audioClockMs = [&]() -> double {
        const qint64 buffered = (qint64)audio.bufferSize() - (qint64)audio.bytesFree();
        const qint64 played = (qint64)audioWritten - buffered;
        double t = (played <= 0) ? audioBaseMs
            : audioBaseMs + played * 1000.0 / (double)bytesPerSec;
        const double dur = totalMs.load();
        if (dur > 0.0 && t > dur) t = dur;
        return t;
    };

    double lastVideoDue = 0.0;
    int    tickCount = 0;
    long   seekCount = 0;

    // ========== ⑦ 用户拖动进度条 ==========
    view.onSeek = [&](double targetMs) {
        const double dur = totalMs.load();
        if (dur <= 0.0) return;

        const double t = qBound(0.0, targetMs, dur - 1.0);
        ++seekCount;

        vdec.requestSeek(t);
        adec.requestSeek(t);

        vq.clear(t);                 // 清空 + 世代 +1 + 记下限
        aq.clear();                  // 清空 + 世代 +1

        // 用 stop + start 重新打开声卡（reset() 会让设备停止消费）
        audio.stop();
        audioDev = audio.start();
        if (!audioDev) qCritical() << "重启声卡失败";

        audioWritten = 0;
        audioBaseMs = t;

        lastVideoDue = t;
        view.setPosition(t);

        qInfo() << "===== seek #" << seekCount << "→" << (qint64)t << "ms =====";
    };

    QTimer timer;
    QObject::connect(&timer, &QTimer::timeout, [&]() {
        const double dur = totalMs.load();
        if (dur > 0.0) view.setDuration(dur);

        // --- 泵音频 ---
        std::vector<uint8_t> chunk;
        while (aq.popIfFits(audio.bytesFree(), chunk)) {
            audioDev->write((const char*)chunk.data(), (qint64)chunk.size());
            audioWritten += chunk.size();
        }

        // --- 视频：追音频时钟 ---
        double vDue = 0.0;
        AVFrame* f = vq.takeLatestDue(audioClockMs(), &vDue);
        if (f) {
            lastVideoDue = vDue;
            view.setPosition(lastVideoDue);
            view.setFrame(f->data[0], f->linesize[0],
                f->data[1], f->linesize[1],
                f->data[2], f->linesize[2],
                f->width, f->height);
            av_frame_free(&f);
        }

        // --- 每 30 次 tick 打印一次同步误差 ---
        if (++tickCount % 30 == 0) {
            const double ac = audioClockMs();
            qInfo() << "音频时钟" << (qint64)ac << "ms"
                << " 视频pts" << (qint64)lastVideoDue << "ms"
                << " 差" << (qint64)(lastVideoDue - ac) << "ms";
        }

        // --- 结束判定 ---
        const bool videoDone = vq.empty() && vq.finished();
        const bool audioDone = aq.empty() && aq.finished()
            && audio.state() != QAudio::ActiveState;
        if (videoDone && audioDone) {
            timer.stop();
            qInfo() << "播放结束，用时" << clock.elapsed() << "ms"
                << "  丢弃视频帧" << vq.dropped()
                << "  送出音频" << audioWritten << "字节";
            QCoreApplication::quit();
        }
        });
    timer.start(10);

    int ret = app.exec();

    vq.stop();
    aq.stop();
    if (vthread.joinable()) vthread.join();
    if (athread.joinable()) athread.join();
    vdec.close();
    adec.close();
    return ret;
}
