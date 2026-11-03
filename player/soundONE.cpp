#define _CRT_SECURE_NO_WARNINGS
#include <cstdio>
#include <cstdint>
#include <vector>

#include <QCoreApplication>
#include <QAudioOutput>
#include <QAudioFormat>
#include <QAudioDeviceInfo>
#include <QIODevice>
#include <QThread>
#include <QDebug>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswresample/swresample.h>
#include <libavutil/opt.h>
#include <libavutil/channel_layout.h>
}

int main(int argc, char* argv[])
{
    QCoreApplication app(argc, argv);

    // ==================== ① 先确定声卡要什么格式 ====================
    QAudioFormat fmt;
    fmt.setSampleRate(48000);
    fmt.setChannelCount(2);
    fmt.setSampleSize(16);
    fmt.setCodec("audio/pcm");
    fmt.setByteOrder(QAudioFormat::LittleEndian);
    fmt.setSampleType(QAudioFormat::SignedInt);

    const QAudioDeviceInfo info = QAudioDeviceInfo::defaultOutputDevice();
    qInfo() << "默认输出设备:" << info.deviceName();

    if (!info.isFormatSupported(fmt)) {
        qWarning() << "不支持 48000/2ch/16bit，改用最接近的:";
        fmt = info.nearestFormat(fmt);
        qWarning() << "  →" << fmt.sampleRate() << "Hz" << fmt.channelCount() << "ch"
            << fmt.sampleSize() << "bit";
    }

    const int      outRate = fmt.sampleRate();
    const int      outCh = fmt.channelCount();
    const AVSampleFormat outFmt = AV_SAMPLE_FMT_S16;

    // ==================== ② 打开声卡 ====================
    QAudioOutput audio(fmt);
    QIODevice* dev = audio.start();          // 返回一个"可写的设备"
    if (!dev) { qCritical() << "打开声卡失败"; return 1; }
    qInfo() << "声卡已打开，缓冲" << audio.bufferSize() << "字节";

    // 往声卡写数据，缓冲满了就等
    auto feed = [&](const unsigned char* data, size_t bytes)
    {
        size_t off = 0;
        while (off < bytes)
        {
            const int freeBytes = audio.bytesFree();
            if (freeBytes <= 0) { QThread::msleep(5); continue; }
            const size_t n = (size_t)qMin<qint64>(freeBytes, (qint64)(bytes - off));
            const qint64 wrote = dev->write((const char*)data + off, (qint64)n);
            if (wrote > 0) off += (size_t)wrote;
            else           QThread::msleep(5);
        }
    };

    // ==================== ③ 打开视频文件，找音频流 ====================
    const char* path = "C:/Users/86150/Videos/2026-07-02 16-02-22.mkv";

    AVFormatContext* mfmt = nullptr;
    if (avformat_open_input(&mfmt, path, nullptr, nullptr) < 0) { qCritical() << "打不开"; return 1; }
    avformat_find_stream_info(mfmt, nullptr);

    const int idx = av_find_best_stream(mfmt, AVMEDIA_TYPE_AUDIO, -1, -1, nullptr, 0);
    if (idx < 0) { qCritical() << "没有音频流"; return 1; }

    const AVCodecParameters* par = mfmt->streams[idx]->codecpar;
    const AVCodec* dec = avcodec_find_decoder(par->codec_id);
    AVCodecContext* ctx = avcodec_alloc_context3(dec);
    avcodec_parameters_to_context(ctx, par);
    if (avcodec_open2(ctx, dec, nullptr) < 0) { qCritical() << "打开解码器失败"; return 1; }

    AVPacket* pkt = av_packet_alloc();
    AVFrame* frame = av_frame_alloc();

    // ==================== ④ 解码 → 转换 → 喂给声卡 ====================
    AVChannelLayout outLayout;
    av_channel_layout_default(&outLayout, outCh);

    SwrContext* swr = nullptr;
    bool swrReady = false;
    uint64_t totalBytes = 0;

    auto handleFrame = [&]()
    {
        if (!swrReady)
        {
            int ret = swr_alloc_set_opts2(&swr,
                &outLayout, outFmt, outRate,
                &frame->ch_layout, (AVSampleFormat)frame->format, frame->sample_rate,
                0, nullptr);
            if (ret < 0 || swr_init(swr) < 0) { qCritical() << "swr 初始化失败"; return; }
            swrReady = true;
            qInfo() << "转换:"
                << frame->sample_rate << "Hz /" << frame->ch_layout.nb_channels << "ch /"
                << av_get_sample_fmt_name((AVSampleFormat)frame->format)
                << "→" << outRate << "Hz /" << outCh << "ch / s16";
        }

        const int maxOut = (int)av_rescale_rnd(
            swr_get_delay(swr, frame->sample_rate) + frame->nb_samples,
            outRate, frame->sample_rate, AV_ROUND_UP);

        std::vector<unsigned char> buf((size_t)maxOut * outCh * 2);
        uint8_t* outPtr = buf.data();

        const int got = swr_convert(swr, &outPtr, maxOut,
            (const uint8_t**)frame->extended_data,
            frame->nb_samples);
        if (got > 0)
        {
            const size_t bytes = (size_t)got * outCh * 2;
            feed(buf.data(), bytes);
            totalBytes += bytes;
        }
    };

    bool flushed = false;
    for (;;)
    {
        int ret = avcodec_receive_frame(ctx, frame);
        if (ret == 0) { handleFrame(); av_frame_unref(frame); continue; }
        if (ret == AVERROR_EOF) break;

        if (av_read_frame(mfmt, pkt) < 0)
        {
            if (flushed) break;
            flushed = true;
            avcodec_send_packet(ctx, nullptr);
            continue;
        }
        if (pkt->stream_index != idx) { av_packet_unref(pkt); continue; }

        ret = avcodec_send_packet(ctx, pkt);
        av_packet_unref(pkt);
        if (ret < 0 && ret != AVERROR(EAGAIN)) continue;
    }

    // ==================== ⑤ 等声卡把剩下的播完 ====================
    qInfo() << "解码结束，共送出" << totalBytes << "字节，等声卡播完…";
    while (audio.state() == QAudio::ActiveState)
        QThread::msleep(20);

    qInfo() << "播放结束";

    if (swr) swr_free(&swr);
    av_frame_free(&frame);
    av_packet_free(&pkt);
    avcodec_free_context(&ctx);
    avformat_close_input(&mfmt);
    return 0;
}