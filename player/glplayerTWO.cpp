#define _CRT_SECURE_NO_WARNINGS
#include <QApplication>
#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <QOpenGLShaderProgram>
#include <QDebug>
#include <QTimer>
#include <QDateTime>
#include <QElapsedTimer>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
#include <libavutil/pixdesc.h>
}

#ifndef GL_UNPACK_ROW_LENGTH
#  define GL_UNPACK_ROW_LENGTH 0x0CF2
#endif
#ifndef GL_LUMINANCE
#  define GL_LUMINANCE 0x1909
#endif

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

class Player
{
public:
    ~Player() { close(); }

    bool open(const char* path)
    {
        if (avformat_open_input(&m_fmt, path, nullptr, nullptr) < 0) return false;
        if (avformat_find_stream_info(m_fmt, nullptr) < 0) return false;

        m_idx = av_find_best_stream(m_fmt, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
        if (m_idx < 0) return false;

        m_tb = m_fmt->streams[m_idx]->time_base;      // ★ 时间基，换算用

        const AVCodecParameters* par = m_fmt->streams[m_idx]->codecpar;
        const AVCodec* dec = avcodec_find_decoder(par->codec_id);
        m_ctx = avcodec_alloc_context3(dec);
        avcodec_parameters_to_context(m_ctx, par);
        if (avcodec_open2(m_ctx, dec, nullptr) < 0) return false;

        m_pkt = av_packet_alloc();
        m_frame = av_frame_alloc();
        return true;
    }

    // 解下一帧。返回 false = 视频播完了
    bool nextFrame()
    {
        m_hasFrame = false;
        for (int guard = 0; guard < 500; ++guard)
        {
            int ret = avcodec_receive_frame(m_ctx, m_frame);
            if (ret == 0)
            {
                if (m_firstPts == AV_NOPTS_VALUE && m_frame->pts != AV_NOPTS_VALUE)
                    m_firstPts = m_frame->pts;        // ★ 记住第一帧的时间
                m_hasFrame = true;
                return true;
            }
            if (ret == AVERROR_EOF) return false;

            if (av_read_frame(m_fmt, m_pkt) < 0)
            {
                if (m_flushed) return false;          // 已经冲刷过，真结束了
                m_flushed = true;
                avcodec_send_packet(m_ctx, nullptr);  // 冲刷
                continue;
            }
            if (m_pkt->stream_index != m_idx) { av_packet_unref(m_pkt); continue; }

            ret = avcodec_send_packet(m_ctx, m_pkt);
            av_packet_unref(m_pkt);
            if (ret < 0 && ret != AVERROR(EAGAIN)) continue;
        }
        return false;
    }

    // ★★★ 当前这一帧「应该在播放的第几毫秒出现」
    double frameTimeMs() const
    {
        if (!m_hasFrame || m_frame->pts == AV_NOPTS_VALUE || m_firstPts == AV_NOPTS_VALUE)
            return 0.0;
        return (m_frame->pts - m_firstPts) * av_q2d(m_tb) * 1000.0;
    }

    bool hasFrame() const { return m_hasFrame; }

    const uint8_t* plane(int i)    const { return m_frame->data[i]; }
    int            linesize(int i) const { return m_frame->linesize[i]; }
    int            width()         const { return m_frame->width; }
    int            height()        const { return m_frame->height; }

    void close()
    {
        if (m_frame) av_frame_free(&m_frame);
        if (m_pkt)   av_packet_free(&m_pkt);
        if (m_ctx)   avcodec_free_context(&m_ctx);
        if (m_fmt)   avformat_close_input(&m_fmt);
    }

private:
    AVFormatContext* m_fmt = nullptr;
    AVCodecContext* m_ctx = nullptr;
    AVFrame* m_frame = nullptr;
    AVPacket* m_pkt = nullptr;
    int              m_idx = -1;
    AVRational       m_tb = { 1, 1000 };
    int64_t          m_firstPts = AV_NOPTS_VALUE;
    bool             m_flushed = false;
    bool             m_hasFrame = false;
};

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
        repaint();
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
        glPixelStorei(GL_UNPACK_ROW_LENGTH, m_linesize[i]);   // ★ 每平面各自的值
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

int main(int argc, char* argv[])
{
    QApplication app(argc, argv);

    Player player;
    if (!player.open("C:/Users/86150/Videos/2026-07-02 16-02-22.mkv")) {
        qCritical() << "打开视频失败";
        return 1;
    }
    player.nextFrame();                       // 先解好第一帧

    View view;
    view.resize(1280, 720);
    view.setWindowTitle("按时间戳播放");
    view.setFrame(player.plane(0), player.linesize(0),
        player.plane(1), player.linesize(1),
        player.plane(2), player.linesize(2),
        player.width(), player.height());
    view.show();

    QElapsedTimer clock;                      // ★ 播放时钟
    clock.start();

    QTimer* timer = new QTimer(&view);
    QObject::connect(timer, &QTimer::timeout, [&]() {

        const double now = clock.elapsed();   // 现在播放到第几毫秒

        // 把「已经到点」的帧一路吃掉，只停在最新的一帧上
        while (player.hasFrame() && player.frameTimeMs() <= now)
        {
            view.setFrame(player.plane(0), player.linesize(0),
                player.plane(1), player.linesize(1),
                player.plane(2), player.linesize(2),
                player.width(), player.height());

            if (!player.nextFrame())           // 播完了
            {
                timer->stop();
                qInfo() << "播放结束，用时" << clock.elapsed() << "ms"
                    << "（视频时长应为 99270 ms）";
                return;
            }
        }
        });
    timer->start(10);                         // ★ 10ms 问一次，不再靠它当节拍器

    return app.exec();
}