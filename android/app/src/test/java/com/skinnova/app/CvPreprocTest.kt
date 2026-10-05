package com.skinnova.app

import com.skinnova.app.ml.CvClassifier
import com.skinnova.app.ml.CvPreprocess
import com.skinnova.app.ml.Rgb
import com.skinnova.app.model.SnJson
import kotlinx.serialization.json.float
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.Assume.assumeTrue
import org.junit.Assert.assertTrue
import org.junit.Test
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs

/** test.md C4: Kotlin preprocessing tensor vs Python's (same decoded pixels) — max |Δ| in pixel units ≤ 1/255. */
class CvPreprocTest {
    private fun res(p: String) = javaClass.classLoader!!.getResource(p)

    @Test fun kotlinMatchesPythonTensor() {
        val idx = res("cv_preproc/index.json")
        assumeTrue("fixture appears after ml/cv/export_tflite.py", idx != null)
        val o = SnJson.parseToJsonElement(idx!!.readText()).jsonObject
        val mean = o["mean"]!!.jsonArray.map { it.jsonPrimitive.float }; val std = o["std"]!!.jsonArray.map { it.jsonPrimitive.float }
        val pre = CvPreprocess(o["size"]!!.jsonPrimitive.int, mean, std, 1.0, emptyList())
        for (im in o["images"]!!.jsonArray) {
            val m = im.jsonObject; val name = m["name"]!!.jsonPrimitive.content
            val w = m["w"]!!.jsonPrimitive.int; val h = m["h"]!!.jsonPrimitive.int
            val raw = res("cv_preproc/$name.rgb")!!.readBytes()
            val px = IntArray(w * h) { i -> ((raw[3 * i].toInt() and 255) shl 16) or ((raw[3 * i + 1].toInt() and 255) shl 8) or (raw[3 * i + 2].toInt() and 255) }
            val got = CvClassifier.preprocess(Rgb(w, h, px), pre)
            val fb = ByteBuffer.wrap(res("cv_preproc/$name.f32")!!.readBytes()).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
            val exp = FloatArray(fb.remaining()).also { fb.get(it) }
            var maxd = 0f
            for (i in exp.indices) maxd = maxOf(maxd, abs(got[i] - exp[i]) * std[i % 3])   // back to [0,1] pixel units
            assertTrue("$name max |Δ| = ${maxd * 255} levels", maxd <= 1f / 255f + 1e-6f)
        }
    }
}
