#define _CRT_SECURE_NO_WARNINGS
#include <QApplication>
#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <QOpenGLShaderProgram>
#include <QDebug>

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
        update();
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

    AVFormatContext* fmt = nullptr;
    avformat_open_input(&fmt,
        "C:/Users/86150/Videos/2026-07-02 16-02-22.mkv", nullptr, nullptr);
    avformat_find_stream_info(fmt, nullptr);

    const int idx = av_find_best_stream(fmt, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    const AVCodecParameters* par = fmt->streams[idx]->codecpar;

    const AVCodec* dec = avcodec_find_decoder(par->codec_id);
    AVCodecContext* ctx = avcodec_alloc_context3(dec);
    avcodec_parameters_to_context(ctx, par);
    avcodec_open2(ctx, dec, nullptr);

    AVPacket* pkt = av_packet_alloc();
    AVFrame* frm = av_frame_alloc();

    while (av_read_frame(fmt, pkt) >= 0)
    {
        if (pkt->stream_index != idx) { av_packet_unref(pkt); continue; }
        int ret = avcodec_send_packet(ctx, pkt);
        av_packet_unref(pkt);
        if (ret < 0 && ret != AVERROR(EAGAIN)) continue;
        if (avcodec_receive_frame(ctx, frm) == 0) break;
    }

    qInfo() << "格式 =" << av_get_pix_fmt_name((AVPixelFormat)frm->format)
        << " 尺寸 =" << frm->width << "x" << frm->height
        << " linesize =" << frm->linesize[0] << frm->linesize[1] << frm->linesize[2];

    View w;
    w.resize(1280, 720);
    w.setWindowTitle("我的第一帧（彩色）");
    w.setFrame(frm->data[0], frm->linesize[0],
        frm->data[1], frm->linesize[1],
        frm->data[2], frm->linesize[2],
        frm->width, frm->height);
    w.show();

    return app.exec();
}