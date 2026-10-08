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
    val model_sha256: String = "", val dataset_rev: String = "", val skin_gate: SkinGate? = null,
    /** Test-time augmentation views averaged as calibrated probabilities (ml/cv/tta_eval.py, adopted by a pre-registered
     *  val rule 2026-10-08): "id", "h" (mirror), "v" (vertical flip), "r180". The skin gate always uses the "id" view. */
    val tta: List<String> = listOf("id"),
)

/** ml/cv/skin_gate.py: p(skin photo) = sigmoid(skin_logit); below [threshold] the photo is probably not skin. */
@Serializable
data class SkinGate(val output: String = "skin_logit", val threshold: Double, val report: String = "")

/** Calibrated scores + p(skin photo) (null with a single-output model). */
data class CvOut(val scores: List<CvScore>, val pSkin: Double?)

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
    fun preprocess(src: Rgb): FloatArray = preprocess(src, pre!!)

    companion object {
        /** Pure: same code path the app runs; JVM-tested against Python (tests/fixtures/cv_preproc, test C4). */
        /** One TTA view of an NHWC [size]×[size]×3 tensor ≡ torch flip(-1) / flip(-2) / both on NCHW (JVM-tested). */
        fun view(x: FloatArray, size: Int, v: String): FloatArray {
            if (v == "id") return x
            val fx = v == "h" || v == "r180"; val fy = v == "v" || v == "r180"
            require(fx || fy) { "unknown TTA view $v" }
            val out = FloatArray(x.size)
            for (y in 0 until size) for (xx in 0 until size) {
                val s = ((if (fy) size - 1 - y else y) * size + (if (fx) size - 1 - xx else xx)) * 3
                val d = (y * size + xx) * 3
                out[d] = x[s]; out[d + 1] = x[s + 1]; out[d + 2] = x[s + 2]
            }
            return out
        }

        fun preprocess(src: Rgb, p: CvPreprocess): FloatArray {
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
    }

    /** (logits, skin logit or null) for one input tensor. Outputs are told apart by shape — TFLite does not guarantee their order. */
    @Synchronized
    private fun runTensor(x: FloatArray): Pair<FloatArray, Float?> {
        val p = pre!!
        val inBuf = ByteBuffer.allocateDirect(x.size * 4).order(ByteOrder.nativeOrder())
        inBuf.asFloatBuffer().put(x)
        val n = interpreter.outputTensorCount
        val logits = Array(1) { FloatArray(p.classes.size) }; val skin = Array(1) { FloatArray(1) }
        val outs = HashMap<Int, Any>()
        for (i in 0 until n) outs[i] = if (interpreter.getOutputTensor(i).shape().last() == 1) skin else logits
        interpreter.runForMultipleInputsOutputs(arrayOf(inBuf), outs)
        return logits[0] to (if (n > 1) skin[0][0] else null)
    }

    /** Single ("id") view: raw logits + skin logit (parity dumps). */
    fun run(src: Rgb): Pair<FloatArray, Float?> = runTensor(preprocess(src))

    fun logits(src: Rgb): FloatArray = run(src).first

    fun classify(src: Rgb): List<CvScore> = classifyWithSkin(src).scores

    /** Calibrated probabilities averaged over the TTA views; p(skin photo) from the "id" view. */
    fun classifyWithSkin(src: Rgb): CvOut {
        val p = pre!!
        val x = preprocess(src)
        var skin: Float? = null
        val avg = DoubleArray(allKeys.size)
        val views = p.tta.ifEmpty { listOf("id") }
        for (v in views) {
            val (lg, sk) = runTensor(view(x, p.size, v))
            if (v == "id") skin = sk
            calibrate(lg).forEach { s -> avg[allKeys.indexOf(s.key)] += s.p / views.size }
        }
        if (skin == null && "id" !in views) skin = runTensor(x).second
        return CvOut(allKeys.mapIndexed { i, k -> CvScore(k, avg[i]) }.sortedByDescending { it.p }, skin?.let { 1.0 / (1.0 + exp(-it.toDouble())) })
    }

    /** False only when the model is fairly sure this is not a photo of skin (soft gate: the user may continue). */
    fun looksLikeSkin(pSkin: Double?): Boolean = pSkin == null || pSkin >= (pre?.skin_gate?.threshold ?: 0.0)

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
