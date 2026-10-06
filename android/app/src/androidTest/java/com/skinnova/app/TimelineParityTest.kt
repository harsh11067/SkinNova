package com.skinnova.app

import android.graphics.BitmapFactory
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.skinnova.app.model.SnJson
import com.skinnova.app.timeline.NoiseFloor
import com.skinnova.app.timeline.Timeline
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.boolean
import kotlinx.serialization.json.double
import kotlinx.serialization.json.doubleOrNull
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import kotlin.math.abs

/**
 * TL3 (test.md §8): Kotlin Timeline ≡ ml/timeline/metrics.py. Fixtures from ml/timeline/make_tl3_fixtures.py: TL1 synthetic
 * pairs + different-spot pairs as lossless PNG, expected metrics computed by Python on the same decoded bytes.
 * align_ok and coin_in_both must agree; on aligned pairs area ratio within 2 % (relative) and contrast delta within 0.5 ΔE.
 */
@RunWith(AndroidJUnit4::class)
class TimelineParityTest {
    private val ctx = InstrumentationRegistry.getInstrumentation().targetContext
    private val testCtx = InstrumentationRegistry.getInstrumentation().context

    @Test fun kotlinMatchesPython() {
        assertTrue("OpenCV failed to load", Timeline.ready)
        val fx = SnJson.parseToJsonElement(testCtx.assets.open("tl3/metrics_cases.json").bufferedReader().readText()).jsonObject
        val tol = fx["tolerance"]!!.jsonObject
        val tolArea = tol["area_ratio_rel"]!!.jsonPrimitive.double; val tolDe = tol["contrast_delta_abs"]!!.jsonPrimitive.double
        val noise = NoiseFloor(0.0, 0.0, fx["noise_floor_n"]!!.jsonPrimitive.content.toInt())
        val fails = ArrayList<String>(); val rows = ArrayList<String>()
        for (e in fx["cases"]!!.jsonArray) {
            val c = e.jsonObject; val id = c["id"]!!.jsonPrimitive.content
            val want = c["expected"]!!.jsonObject
            fun bmp(k: String) = testCtx.assets.open("tl3/" + c[k]!!.jsonPrimitive.content).use { BitmapFactory.decodeStream(it) }
            val seed = c["seed_xy"]!!.jsonArray.map { it.jsonPrimitive.double }
            val coinMm = c["coin_mm"]?.jsonPrimitive?.doubleOrNull   // JsonNull → null
            val t = System.nanoTime()
            val got = Timeline.compute(bmp("base"), bmp("new"), seed[0], seed[1], coinMm, null, null, noise, true)
            val ms = (System.nanoTime() - t) / 1e6
            val wAlign = want["align_ok"]!!.jsonPrimitive.boolean
            val dArea = rel(got.areaRatio, num(want, "area_ratio")); val dDe = absDiff(got.contrastDelta, num(want, "contrast_delta"))
            if (got.alignOk != wAlign) fails += "$id align_ok ${got.alignOk} vs $wAlign"
            else if (wAlign) {
                if (dArea == null || dArea > tolArea) fails += "$id area_ratio ${got.areaRatio} vs ${num(want, "area_ratio")}"
                if (dDe == null || dDe > tolDe) fails += "$id contrast_delta ${got.contrastDelta} vs ${num(want, "contrast_delta")}"
                if (got.coinInBoth != want["coin_in_both"]!!.jsonPrimitive.boolean) fails += "$id coin_in_both ${got.coinInBoth}"
            }
            rows += """{"id":"$id","align_ok":${got.alignOk},"align_ok_py":$wAlign,"inliers":${got.alignInliers},""" +
                """"inliers_py":${want["align_inliers"]},"area_ratio":${got.areaRatio},"area_ratio_py":${want["area_ratio"]},""" +
                """"contrast_delta":${got.contrastDelta},"contrast_delta_py":${want["contrast_delta"]},"coin_in_both":${got.coinInBoth},""" +
                """"coin_in_both_py":${want["coin_in_both"]},"seg_ok":${got.segOk},"seg_ok_py":${want["seg_ok"]},""" +
                """"confidence":"${got.confidence}","confidence_py":${want["confidence"]},"ms":${"%.0f".format(ms)}}"""
        }
        File(ctx.getExternalFilesDir("bench"), "tl3_parity_device.json").writeText(
            """{"n":${rows.size},"failures":${fails.size},"fail_list":[${fails.joinToString(",") { "\"$it\"" }}],"rows":[${rows.joinToString(",")}]}""")
        assertTrue("TL3 mismatches (${fails.size}/${rows.size}):\n" + fails.joinToString("\n"), fails.isEmpty())
    }

    private fun num(o: JsonObject, k: String) = o[k]?.jsonPrimitive?.doubleOrNull
    private fun rel(a: Double?, b: Double?) = if (a == null || b == null || b == 0.0) null else abs(a / b - 1)
    private fun absDiff(a: Double?, b: Double?) = if (a == null || b == null) null else abs(a - b)
}
