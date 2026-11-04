#define _CRT_SECURE_NO_WARNINGS
#include <QApplication>
#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <QOpenGLShaderProgram>
#include <QDebug>
#include <QTimer>
#include <QElapsedTimer>
#include <thread>
#include <mutex>
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
// ==================== ① 帧队列 ====================
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
    // 解码线程：放货。队列满了就睡着等（背压）
    bool push(AVFrame* f, double dueMs)
    {
        std::unique_lock<std::mutex> lk(m_mtx);
        m_cv.wait(lk, [this] { return m_q.size() < m_max || m_stop; });
        if (m_stop) { av_frame_free(&f); return false; }
        m_q.push_back({ f, dueMs });
        m_cv.notify_all();
        return true;
    }
    // 显示端：把到点的都取走，只留最新的，更旧的释放（这就是丢帧）
    // ★ ⑥-2：outDueMs 顺便把「这一帧该在几毫秒」带出去
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
    bool                    m_stop = false;       // 程序要退出
    bool                    m_finished = false;   // 解码已经到结尾
    long                    m_dropped = 0;
};
// ==================== ② 解码器（跑在解码线程里） ====================
class Decoder
{
public:
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
    // 解一帧，"搬"到新的 AVFrame 上返回；调用者负责 av_frame_free
    AVFrame* nextOwnedFrame()
    {
        for (int guard = 0; guard < 500; ++guard)
        {
            int ret = avcodec_receive_frame(m_ctx, m_tmp);
            if (ret == 0)
            {
                if (m_firstPts == AV_NOPTS_VALUE && m_tmp->pts != AV_NOPTS_VALUE)
                    m_firstPts = m_tmp->pts;
                AVFrame* out = av_frame_alloc();
                if (!out) return nullptr;
                av_frame_move_ref(out, m_tmp);        // 所有权转移
                return out;
            }
            if (ret == AVERROR_EOF) return nullptr;
            if (av_read_frame(m_fmt, m_pkt) < 0)
            {
                if (m_flushed) return nullptr;
                m_flushed = true;
                avcodec_send_packet(m_ctx, nullptr);
                continue;
            }
            if (m_pkt->stream_index != m_idx) { av_packet_unref(m_pkt); continue; }
            ret = avcodec_send_packet(m_ctx, m_pkt);
            av_packet_unref(m_pkt);
            if (ret < 0 && ret != AVERROR(EAGAIN)) continue;
        }
        return nullptr;
    }
    double dueMs(int64_t pts) const
    {
        if (pts == AV_NOPTS_VALUE || m_firstPts == AV_NOPTS_VALUE) return 0.0;
        return (pts - m_firstPts) * av_q2d(m_tb) * 1000.0;
    }
    void close()
    {
        if (m_tmp) av_frame_free(&m_tmp);
        if (m_pkt) av_packet_free(&m_pkt);
        if (m_ctx) avcodec_free_context(&m_ctx);
        if (m_fmt) avformat_close_input(&m_fmt);
    }
private:
    AVFormatContext* m_fmt = nullptr;
    AVCodecContext* m_ctx = nullptr;
    AVFrame* m_tmp = nullptr;
    AVPacket* m_pkt = nullptr;
    int              m_idx = -1;
    AVRational       m_tb = { 1, 1000 };
    int64_t          m_firstPts = AV_NOPTS_VALUE;
    bool             m_flushed = false;
};
// ==================== ③ View ====================
class View : public QOpenGLWidget, protected QOpenGLFunctions
{
public:
    void setFrame(const uint8_t* y, int lsY,
        const uint8_t* u, int lsU,
        const uint8_t* v, int lsV,
        int w, int h)
    {
        m_plane[0] = y;  m_linesize[0] = lsY;
        m_plane[1] = u;  m_linesize[1] = lsU;
        m_plane[2] = v;  m_linesize[2] = lsV;
        m_w = w;  m_h = h;
        repaint();                     // 必须同步画完，外面随后要释放帧
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
        glActiveTexture(GL_TEXTURE0);
    }
private:
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
    QOpenGLShaderProgram* m_prog = nullptr;
    GLuint         m_tex[3] = { 0, 0, 0 };
    const uint8_t* m_plane[3] = { nullptr, nullptr, nullptr };
    int            m_linesize[3] = { 0, 0, 0 };
    int            m_w = 0, m_h = 0;
};
// ==================== ④ PCM 队列（音频用） ====================
class PcmQueue
{
public:
    explicit PcmQueue(size_t maxBytes = 96000) : m_max(maxBytes) {}   // 约 0.5 秒
    ~PcmQueue()
    {
        stop();
        std::lock_guard<std::mutex> lk(m_mtx);
        m_q.clear();
    }
    bool push(std::vector<uint8_t>&& chunk)
    {
        std::unique_lock<std::mutex> lk(m_mtx);
        m_cv.wait(lk, [this] { return m_bytes < m_max || m_stop; });
        if (m_stop) return false;
        m_bytes += chunk.size();
        m_q.push_back(std::move(chunk));
        m_cv.notify_all();
        return true;
    }
    // 如果队头这一块能塞进 freeBytes，就取出来；否则返回 false
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
    bool                               m_stop = false;
    bool                               m_finished = false;
};
// ==================== ⑤ 音频解码器 ====================
class AudioDecoder
{
public:
    bool open(const char* path, int outRate, int outCh)
    {
        if (avformat_open_input(&m_fmt, path, nullptr, nullptr) < 0) return false;
        if (avformat_find_stream_info(m_fmt, nullptr) < 0) return false;
        m_idx = av_find_best_stream(m_fmt, AVMEDIA_TYPE_AUDIO, -1, -1, nullptr, 0);
        if (m_idx < 0) return false;
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
    // 解一块 PCM 出来（s16 交错）。返回 false = 播完了
    bool nextChunk(std::vector<uint8_t>& out)
    {
        out.clear();
        for (int guard = 0; guard < 500; ++guard)
        {
            int ret = avcodec_receive_frame(m_ctx, m_frame);
            if (ret == 0) { convert(out); av_frame_unref(m_frame); return true; }
            if (ret == AVERROR_EOF) return false;
            if (av_read_frame(m_fmt, m_pkt) < 0)
            {
                if (m_flushed) return false;
                m_flushed = true;
                avcodec_send_packet(m_ctx, nullptr);
                continue;
            }
            if (m_pkt->stream_index != m_idx) { av_packet_unref(m_pkt); continue; }
            ret = avcodec_send_packet(m_ctx, m_pkt);
            av_packet_unref(m_pkt);
            if (ret < 0 && ret != AVERROR(EAGAIN)) continue;
        }
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
    const char* path = "https://demo.unified-streaming.com/k8s/live/stable/scte35.isml/.m3u8";
    // ========== ③ 窗口 ==========
    View view;
    view.resize(1280, 720);
    view.setWindowTitle("⑥-2：音频时钟 vs 视频 pts");
    view.show();
    clock.start();
    // ========== ④ 视频解码线程 ==========
    std::thread vthread([&]() {
        if (!vdec.open(path)) { qWarning() << "视频打不开"; vq.markFinished(); return; }
        for (;;) {
            AVFrame* f = vdec.nextOwnedFrame();
            if (!f) break;
            if (!vq.push(f, vdec.dueMs(f->pts))) break;
        }
        qInfo() << "[视频] 解码线程结束";
        vq.markFinished();
        });
    // ========== ⑤ 音频解码线程 ==========
    std::thread athread([&]() {
        if (!adec.open(path, outRate, outCh)) { qWarning() << "音频打不开"; aq.markFinished(); return; }
        for (;;) {
            std::vector<uint8_t> chunk;
            if (!adec.nextChunk(chunk)) break;
            if (chunk.empty()) continue;
            if (!aq.push(std::move(chunk))) break;
        }
        qInfo() << "[音频] 解码线程结束";
        aq.markFinished();
        });
    // ========== ⑥ 界面定时器：画视频 + 泵音频 ==========
    uint64_t audioWritten = 0;
    // ★ ⑥-2：音频时钟 —— 声卡真正播出去了多少毫秒
    const int bytesPerSec = outRate * outCh * 2;
    auto audioClockMs = [&]() -> double {
        const qint64 buffered = (qint64)audio.bufferSize() - (qint64)audio.bytesFree();
        const qint64 played = (qint64)audioWritten - buffered;
        if (played <= 0) return 0.0;
        return played * 1000.0 / (double)bytesPerSec;
    };
    double lastVideoDue = 0.0;      // 当前画面是「第几毫秒的帧」
    int    tickCount = 0;
    QTimer timer;
    QObject::connect(&timer, &QTimer::timeout, [&]() {
        const double now = clock.elapsed();
        // --- 泵音频：能塞多少塞多少 ---
        std::vector<uint8_t> chunk;
        while (aq.popIfFits(audio.bytesFree(), chunk)) {
            audioDev->write((const char*)chunk.data(), (qint64)chunk.size());
            audioWritten += chunk.size();
        }
        // --- 视频：按自身时钟对表（⑥-3 会改成追音频） ---
        double vDue = 0.0;
        AVFrame* f = vq.takeLatestDue(audioClockMs(), &vDue);   // ★ 视频追音频
        if (f) {
            lastVideoDue = vDue;
            view.setFrame(f->data[0], f->linesize[0],
                f->data[1], f->linesize[1],
                f->data[2], f->linesize[2],
                f->width, f->height);
            av_frame_free(&f);
        }
        // --- ★ ⑥-2：每 30 次 tick（约 300ms）打印一次两个时钟的差 ---
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