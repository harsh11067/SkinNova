package com.skinnova.app.ml

import android.annotation.SuppressLint
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.isActive
import kotlinx.coroutines.withContext
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.coroutines.coroutineContext
import kotlin.math.sqrt

/** contracts §9 audio: WAV, 16 kHz, mono, PCM16, ≤ 30 s. Pure functions in [Wav] are JVM-tested (test.md V1). */
object Wav {
    const val RATE = 16_000
    const val MAX_SECONDS = 30

    fun encode(pcm: ShortArray, rate: Int = RATE): ByteArray {
        val data = pcm.size * 2
        val b = ByteBuffer.allocate(44 + data).order(ByteOrder.LITTLE_ENDIAN)
        b.put("RIFF".toByteArray()); b.putInt(36 + data); b.put("WAVE".toByteArray())
        b.put("fmt ".toByteArray()); b.putInt(16); b.putShort(1); b.putShort(1); b.putInt(rate); b.putInt(rate * 2); b.putShort(2); b.putShort(16)
        b.put("data".toByteArray()); b.putInt(data)
        pcm.forEach { b.putShort(it) }
        return b.array()
    }

    /** Energy VAD: drop leading/trailing 30 ms frames whose RMS < max(threshold, 0.1 × loudest frame); keep 150 ms margin. */
    fun trimSilence(pcm: ShortArray, rate: Int = RATE, absThreshold: Double = 300.0): ShortArray {
        val frame = rate * 30 / 1000
        val n = pcm.size / frame
        if (n == 0) return pcm
        val rms = DoubleArray(n) { f ->
            var s = 0.0; for (i in f * frame until (f + 1) * frame) s += pcm[i].toDouble() * pcm[i]
            sqrt(s / frame)
        }
        val thr = maxOf(absThreshold, 0.1 * (rms.maxOrNull() ?: 0.0))
        val first = rms.indexOfFirst { it >= thr }
        if (first < 0) return ShortArray(0)
        val last = rms.indexOfLast { it >= thr }
        val margin = 5
        val s = maxOf(0, first - margin) * frame
        val e = minOf(pcm.size, (last + 1 + margin) * frame)
        return pcm.copyOfRange(s, e)
    }

    fun cap(pcm: ShortArray, rate: Int = RATE) = if (pcm.size > rate * MAX_SECONDS) pcm.copyOf(rate * MAX_SECONDS) else pcm
}

class AudioCapture {
    @Volatile private var stopRequested = false

    fun stop() { stopRequested = true }

    /** Records until [stop] or 30 s. onLevel gets 0..1 for the waveform; onElapsed in ms. Returns trimmed PCM16 WAV bytes. */
    @SuppressLint("MissingPermission")   // caller checks RECORD_AUDIO
    suspend fun record(onLevel: (Float) -> Unit, onElapsed: (Long) -> Unit): ByteArray = withContext(Dispatchers.IO) {
        stopRequested = false
        val minBuf = AudioRecord.getMinBufferSize(Wav.RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
        val rec = AudioRecord(MediaRecorder.AudioSource.VOICE_RECOGNITION, Wav.RATE, AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT, maxOf(minBuf, Wav.RATE / 5 * 2))
        val all = ByteArrayOutputStream()
        val chunk = ShortArray(Wav.RATE / 20)
        var total = 0
        try {
            rec.startRecording()
            while (coroutineContext.isActive && !stopRequested && total < Wav.RATE * Wav.MAX_SECONDS) {
                val n = rec.read(chunk, 0, chunk.size)
                if (n <= 0) continue
                var s = 0.0
                for (i in 0 until n) { s += chunk[i].toDouble() * chunk[i]; all.write(chunk[i].toInt() and 0xFF); all.write((chunk[i].toInt() shr 8) and 0xFF) }
                total += n
                onLevel((sqrt(s / n) / 6000.0).toFloat().coerceIn(0f, 1f))
                onElapsed(total * 1000L / Wav.RATE)
            }
        } finally {
            runCatching { rec.stop() }; rec.release()
        }
        val bytes = all.toByteArray()
        val pcm = ShortArray(bytes.size / 2) { i -> ((bytes[2 * i].toInt() and 0xFF) or (bytes[2 * i + 1].toInt() shl 8)).toShort() }
        Wav.encode(Wav.trimSilence(Wav.cap(pcm)))
    }
}
