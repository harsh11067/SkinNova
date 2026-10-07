package com.skinnova.app

import android.graphics.BitmapFactory
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.skinnova.app.ml.AnalysisPipeline
import com.skinnova.app.ml.GemmaEngine
import com.skinnova.app.ml.Llm
import com.skinnova.app.ml.LlmTask
import com.skinnova.app.ml.QualityGate
import com.skinnova.app.ml.toRgb
import com.skinnova.app.model.Mode
import com.skinnova.app.model.QuestionnaireAnswers
import com.skinnova.app.model.SnJson
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.json.double
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import kotlin.math.abs

/**
 * On-device tests (test.md C5, A1-style pipeline, A8, L7/A2 bench). Run with the phone connected:
 *   ./gradlew connectedOfflineDebugAndroidTest
 */
@RunWith(AndroidJUnit4::class)
class DeviceTests {
    private val ctx = InstrumentationRegistry.getInstrumentation().targetContext
    private val testCtx = InstrumentationRegistry.getInstrumentation().context
    private val c get() = (ctx.applicationContext as SkinNovaApp).container

    /** C5: 20 fixture images on the phone vs the desktop .tflite: top-1 20/20, max |Δp| ≤ 0.02. */
    @Test fun cvOnDeviceParity() {
        assumeTrue("CV model not bundled yet", c.cv.available)
        val exp = SnJson.parseToJsonElement(testCtx.assets.open("cv_fixtures/expected.json").bufferedReader().readText()).jsonArray
        var top1 = 0; var maxd = 0.0
        for (e in exp) {
            val o = e.jsonObject
            val bmp = testCtx.assets.open("cv_fixtures/" + o["file"]!!.jsonPrimitive.content).use { BitmapFactory.decodeStream(it) }
            val out = c.cv.classifyWithSkin(bmp.toRgb()); val got = out.scores.associate { it.key to it.p }
            o["skin"]?.jsonPrimitive?.double?.let { want -> maxd = maxOf(maxd, abs((out.pSkin ?: -1.0) - want)) }   // skin gate parity too
            val want = o["probs"]!!.jsonObject.mapValues { it.value.jsonPrimitive.double }
            if (got.maxBy { it.value }.key == want.maxBy { it.value }.key) top1++
            want.forEach { (k, v) -> maxd = maxOf(maxd, abs((got[k] ?: 0.0) - v)) }
        }
        File(ctx.getExternalFilesDir("bench"), "cv_parity_device.json").writeText("""{"n":${exp.size},"top1_match":$top1,"max_abs_dprob":$maxd}""")
        assertEquals(exp.size, top1)
        assertTrue("max |Δp| = $maxd", maxd <= 0.02)
    }

    /** A1-style: real CV + real rules + Basic-mode path when the LLM is unavailable — no crash, red flags kept. */
    @Test fun pipelineBasicModeOnDevice() = runBlocking {
        assumeTrue(c.cv.available)
        val bmp = testCtx.assets.open("cv_fixtures/00.jpg").use { BitmapFactory.decodeStream(it) }
        val cv = c.cv.classify(bmp.toRgb())
        val noLlm = object : Llm {
            override val available = false
            override suspend fun generate(task: LlmTask, system: String, user: String, imagePath: String?, audio: ByteArray?, onToken: (String) -> Unit) = error("unused")
        }
        val ans = QuestionnaireAnswers("face", "1_6m", 0, 2, "growing", true, false, false, false, "lt_12")
        val r = AnalysisPipeline(c.labels, c.prompts, c.parser, c.guards, noLlm).run(ans, cv, false, null, "en") {}
        assertEquals(Mode.basic, r.mode)
        assertTrue("R5 child rule must fire", "rf_r5" in r.ruleMessages)
        assertTrue(com.skinnova.app.model.Tier.parse(r.finalTier)!! >= com.skinnova.app.model.Tier.MODERATE)
        assertTrue(QualityGate.check(bmp.toRgb()).minSide > 0)
    }

    /** A8: stored photos are AES-GCM ciphertext (not a JPEG), and round-trip correctly. */
    @Test fun imagesAreEncryptedAtRest() = runBlocking {
        val bmp = testCtx.assets.open("cv_fixtures/00.jpg").use { BitmapFactory.decodeStream(it) }
        val ref = c.images.save(bmp)
        val raw = c.images.rawFile(ref).readBytes()
        assertFalse("ciphertext must not start with the JPEG SOI marker", raw[0] == 0xFF.toByte() && raw[1] == 0xD8.toByte())
        val back = c.images.load(ref)!!
        assertEquals(bmp.width, back.width)
        c.images.delete(ref)
        assertFalse(c.images.rawFile(ref).exists())
    }

    /** L7 / A2: real Gemma engine on the phone (needs the .litertlm imported or adb-pushed). Writes bench/outputs.jsonl. */
    @Test fun realEngineBench() = runBlocking {
        val model = c.models.scanAndVerify()
        assumeTrue("no verified .litertlm on the device", model != null)
        val bmp = testCtx.assets.open("cv_fixtures/00.jpg").use { BitmapFactory.decodeStream(it) }
        val img = File(ctx.cacheDir, "bench.jpg").also { f -> f.outputStream().use { bmp.compress(android.graphics.Bitmap.CompressFormat.JPEG, 90, it) } }
        val cv = c.cv.classify(bmp.toRgb())
        val ans = QuestionnaireAnswers("arm", "1_4w", 2, 0, "spreading", false, false, true, false, "18_39")
        val initsBefore = c.engineHolder.initCount
        val llm = c.llm as GemmaEngine
        llm.timeoutMs = 600_000L   // measure the real time; the app itself gives up at 180 s (→ Basic mode)
        val out = File(ctx.getExternalFilesDir("bench"), "outputs.jsonl")
        val warm = try {
            // run 1 = cold (engine load + first-run caches), run 2 = warm (the app preloads the engine at start: what users see)
            (1..2).map { run ->
                val t0 = System.nanoTime(); var ttft = -1L
                val r = c.pipeline().run(ans, cv, false, img.path, "en") { s -> if (ttft < 0 && s is com.skinnova.app.ml.AnalysisState.Generating) ttft = System.nanoTime() }
                val total = (System.nanoTime() - t0) / 1e9
                out.appendText("""{"model":"${model!!.id}","run":"${if (run == 1) "cold" else "warm"}","backend":"${c.engineHolder.backendName}","mode":"${r.mode}","fallback":"${r.fallbackReason}","total_s":$total,"ttft_s":${if (ttft > 0) (ttft - t0) / 1e9 else -1},"engine_init_s":${c.engineHolder.lastInitMs / 1000.0},"tier":"${r.finalTier}","engine_inits":${c.engineHolder.initCount}}""" + "\n")
                total
            }.last()
        } finally { llm.timeoutMs = 180_000L }
        assertTrue("one engine per process (no re-init per analysis)", c.engineHolder.initCount - initsBefore <= 1)
        val budget = if (c.engineHolder.backendName == "GPU") 60.0 else 120.0
        assertTrue("S8 ${c.engineHolder.backendName} budget $budget s, warm run took $warm s", warm <= budget)
    }

    /** A9: GPU-failure path — with the CPU backend forced, a full (LLM) result is still produced. */
    @Test fun cpuFallbackProducesResult() = runBlocking {
        val model = c.models.scanAndVerify()
        assumeTrue("no verified .litertlm on the device", model != null)
        val bmp = testCtx.assets.open("cv_fixtures/00.jpg").use { BitmapFactory.decodeStream(it) }
        val img = File(ctx.cacheDir, "a9.jpg").also { f -> f.outputStream().use { bmp.compress(android.graphics.Bitmap.CompressFormat.JPEG, 95, it) } }
        val ans = QuestionnaireAnswers("arm", "1_4w", 2, 0, "spreading", false, false, true, false, "18_39")
        c.engineHolder.forceBackend("CPU")
        try {
            val r = c.pipeline().run(ans, c.cv.classify(bmp.toRgb()), false, img.path, "en") {}
            assertEquals("CPU", c.engineHolder.backendName)
            assertEquals("fallback reason: ${r.fallbackReason}", Mode.full, r.mode)
        } finally {
            c.engineHolder.forceBackend("GPU")
        }
    }
}
