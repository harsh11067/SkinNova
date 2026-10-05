package com.skinnova.app.ml

import android.content.Context
import android.graphics.Bitmap
import com.skinnova.app.model.CvScore
import com.skinnova.app.model.SnJson
import kotlinx.serialization.Serializable
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.exp

/** assets/cv/preprocess.json — written by ml/cv/export_tflite.py together with the model. */
@Serializable
data class CvPreprocess(
    val size: Int, val mean: List<Float>, val std: List<Float>, val temperature: Double,
    val classes: List<String>, val layout: String = "NHWC", val normalize_long_side: Int = 512,
    val model_sha256: String = "", val dataset_rev: String = "",
)

/**
 * LiteRT .tflite classifier (architecture §2). Preprocessing ≡ ml/cv/dataset.py eval_transform via ImageOps (PIL port).
 * Output: calibrated probabilities (softmax(logits / T)) for every label key; classes the CV model doesn't know get 0.
 */
class CvClassifier(private val ctx: Context, private val allKeys: List<String>) {
    val pre: CvPreprocess? by lazy {
        runCatching { SnJson.decodeFromString(CvPreprocess.serializer(), ctx.assets.open("cv/preprocess.json").bufferedReader().readText()) }.getOrNull()
    }
    val available: Boolean get() = pre != null && runCatching { ctx.assets.openFd("cv/skin_cls.tflite").close() }.isSuccess

    private val interpreter: Interpreter by lazy {
        val fd = ctx.assets.openFd("cv/skin_cls.tflite")
        val buf: MappedByteBuffer = FileInputStream(fd.fileDescriptor).channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
        Interpreter(buf, Interpreter.Options().setNumThreads(4))
    }

    /** Float NHWC input tensor (exposed for the C4 parity dump). */
    fun preprocess(src: Rgb): FloatArray {
        val p = pre!!
        val img = ImageOps.cvInput(ImageOps.normalizeLongSide(src, p.normalize_long_side), p.size)
        val out = FloatArray(p.size * p.size * 3)
        for (i in img.px.indices) {
            val v = img.px[i]
            out[i * 3] = (((v shr 16) and 0xFF) / 255f - p.mean[0]) / p.std[0]
            out[i * 3 + 1] = (((v shr 8) and 0xFF) / 255f - p.mean[1]) / p.std[1]
            out[i * 3 + 2] = ((v and 0xFF) / 255f - p.mean[2]) / p.std[2]
        }
        return out
    }

    @Synchronized
    fun logits(src: Rgb): FloatArray {
        val p = pre!!
        val x = preprocess(src)
        val inBuf = ByteBuffer.allocateDirect(x.size * 4).order(ByteOrder.nativeOrder())
        inBuf.asFloatBuffer().put(x)
        val out = Array(1) { FloatArray(p.classes.size) }
        interpreter.run(inBuf, out)
        return out[0]
    }

    fun classify(src: Rgb): List<CvScore> = calibrate(logits(src))

    fun calibrate(logits: FloatArray): List<CvScore> {
        val p = pre!!
        val z = logits.map { it / p.temperature }
        val m = z.max()
        val e = z.map { exp(it - m) }
        val s = e.sum()
        val probs = p.classes.zip(e.map { it / s }).toMap()
        return allKeys.map { CvScore(it, probs[it] ?: 0.0) }.sortedByDescending { it.p }
    }
}

fun Bitmap.toRgb(): Rgb {
    val px = IntArray(width * height)
    getPixels(px, 0, width, 0, 0, width, height)
    for (i in px.indices) px[i] = px[i] and 0xFFFFFF
    return Rgb(width, height, px)
}

fun Rgb.toBitmap(): Bitmap = Bitmap.createBitmap(IntArray(px.size) { px[it] or (0xFF shl 24) }, w, h, Bitmap.Config.ARGB_8888)
