package com.skinnova.app

import android.app.Application
import android.graphics.BitmapFactory
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onAllNodesWithText
import androidx.compose.ui.test.onRoot
import androidx.compose.ui.test.captureToImage
import androidx.compose.ui.graphics.asAndroidBitmap
import androidx.compose.ui.test.printToString
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.skinnova.app.ml.AnalysisPipeline
import com.skinnova.app.ml.AnalysisState
import com.skinnova.app.ml.GemmaEngine
import com.skinnova.app.ml.Llm
import com.skinnova.app.ml.LlmTask
import com.skinnova.app.model.SnJson
import com.skinnova.app.model.Tier
import com.skinnova.app.ui.Draft
import com.skinnova.app.ui.EtaEstimator
import com.skinnova.app.ui.SessionViewModel
import com.skinnova.app.ui.TimelineViewModel
import com.skinnova.app.ui.theme.SkinNovaTheme
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

private val app get() = ApplicationProvider.getApplicationContext<Application>() as SkinNovaApp
private fun s(id: Int) = app.getString(id)

/** A fake Gemma that writes slowly (≈ 10 s), token by token, like the phone. */
private class SlowLlm(private val reply: String, private val ms: Long = 10_000) : Llm {
    override val available = true
    override suspend fun generate(task: LlmTask, system: String, user: String, imagePath: String?, audio: ByteArray?, onToken: (String) -> Unit): String {
        onToken(""); delay(1_000)
        val steps = 30
        for (i in 1..steps) { delay((ms - 1_000) / steps); onToken(reply.take(reply.length * i / steps)) }
        return reply
    }
}

private fun shot(compose: androidx.compose.ui.test.junit4.ComposeContentTestRule, name: String) {
    compose.waitForIdle()
    val b = compose.onRoot().captureToImage().asAndroidBitmap()
    File(app.getExternalFilesDir("screens"), "$name.png").also { it.parentFile?.mkdirs() }.outputStream().use { b.compress(android.graphics.Bitmap.CompressFormat.PNG, 100, it) }
}

/** v2.1: the one-time optimising screen (real engine load; screenshot only). */
@RunWith(AndroidJUnit4::class)
class OptimizeScreenShot {
    @get:Rule val compose = createComposeRule()
    @Test fun optimizeScreen() {
        val vm = SessionViewModel(app)
        compose.setContent { SkinNovaTheme(app.container.settings.dark.value) { com.skinnova.app.ui.screens.OptimizeScreen(vm) {} } }
        Thread.sleep(4_000); shot(compose, "22_optimizing")
    }
}

/** v2.1 item 2 on the phone: result first (≤ 2 s), the full result swaps in, the advice level never drops. */
@RunWith(AndroidJUnit4::class)
class ProgressiveResultTest {
    @get:Rule val compose = createComposeRule()

    @Test fun resultFirstThenSwap() = runBlocking {
        assumeTrue(app.container.cv.available)
        val c = app.container; val realLlm = c.llm; val history0 = c.settings.history.value
        c.settings.setHistory(false)   // never write a test result into a tester's history
        c.llm = SlowLlm(FAKE_ANALYSIS_V21)
        try {
            val vm = SessionViewModel(app)
            val bmp = InstrumentationRegistry.getInstrumentation().context.assets.open("cv_fixtures/00.jpg").use { BitmapFactory.decodeStream(it) }
            InstrumentationRegistry.getInstrumentation().runOnMainSync {
                vm.setPhoto(bmp); vm.update { Draft("arm", "1_4w", 2, 0, "no", false, false, false, false, "18_39") }
            }
            withTimeout(20_000) { vm.cv.first { it.isNotEmpty() } }
            compose.setContent { SkinNovaTheme(c.settings.dark.value) { App(vm, TimelineViewModel(app), startRoute = Routes.ANALYZING) } }
            val t0 = System.currentTimeMillis()
            InstrumentationRegistry.getInstrumentation().runOnMainSync { vm.analyze() }
            try {
                compose.waitUntil(5_000) { compose.onAllNodesWithText(s(R.string.res_title)).fetchSemanticsNodes().isNotEmpty() }
            } catch (e: Throwable) {
                throw AssertionError("no result screen; state=${vm.state.value::class.simpleName} result=${vm.result.value?.fallbackReason} pending=${vm.pending.value}\n" +
                    compose.onRoot(useUnmergedTree = true).printToString(25), e)
            }
            val firstMs = System.currentTimeMillis() - t0
            compose.waitUntil(8_000) { (vm.progress.value?.fraction ?: 0f) > 0.3f }   // mid-writing: banner with progress
            shot(compose, "20_result_first_writing")
            assertTrue("banner while writing", compose.onAllNodesWithText(s(R.string.wb_title)).fetchSemanticsNodes().isNotEmpty())
            val early = vm.result.value!!
            assertTrue(early.fallbackReason == AnalysisPipeline.PENDING)
            val end = withTimeout(60_000) { vm.state.first { it is AnalysisState.Done } } as AnalysisState.Done
            compose.waitUntil(5_000) { compose.onAllNodesWithText(s(R.string.wb_title)).fetchSemanticsNodes().isEmpty() }
            val full = end.result
            shot(compose, "21_result_swapped")
            assertTrue("tier never drops", Tier.parse(full.finalTier)!! >= Tier.parse(early.finalTier)!!)
            assertTrue(full.mode == com.skinnova.app.model.Mode.full)
            File(app.getExternalFilesDir("bench"), "progressive.json").writeText("""{"first_result_ms":$firstMs,"early_tier":"${early.finalTier}","final_tier":"${full.finalTier}","early_top":"${early.output.possibleCategories.first().key}","final_top":"${full.output.possibleCategories.first().key}"}""")
            assertTrue("first result in $firstMs ms", firstMs <= 2_000)
        } finally { c.llm = realLlm; c.settings.setHistory(history0) }
    }
}

/**
 * v2.1 items 1, 3, 4 with the real model on the phone.
 *  - memoryBench (-e tokens 4096|2048): peak PSS while one analysis runs (item 4).
 *  - greedyParity: 3 frozen llm_val prompts (androidTest assets l7/cases.json, PC greedy outputs at 2,048) generated greedily
 *    on the phone; first case repeated (same input → same text); records the token timeline to score the ETA (item 3).
 */
@RunWith(AndroidJUnit4::class)
class V21DeviceBench {
    private val c get() = app.container
    private val out get() = File(app.getExternalFilesDir("bench"), "v21.jsonl")

    @Test fun memoryBench() = runBlocking {
        assumeTrue(c.models.scanAndVerify() != null)
        val tokens = InstrumentationRegistry.getArguments().getString("tokens")?.toInt() ?: 2048
        c.engineHolder.setMaxTokens(tokens)
        var peak = 0L; var sampling = true
        val sampler = Thread { while (sampling) { peak = maxOf(peak, android.os.Debug.getPss()); Thread.sleep(400) } }.apply { start() }
        val ctx = InstrumentationRegistry.getInstrumentation().context
        val case = SnJson.parseToJsonElement(ctx.assets.open("l7/cases.json").bufferedReader().readText()).jsonArray[0].jsonObject
        c.engineHolder.use { }   // load first (not timed into the generation)
        val t0 = System.nanoTime(); var first = 0L
        val text = c.llm.generate(LlmTask.ANALYZE, case["system"]!!.jsonPrimitive.content, case["user"]!!.jsonPrimitive.content) { t ->
            if (first == 0L && t.isNotEmpty()) first = System.nanoTime() }
        val totalMs = (System.nanoTime() - t0) / 1e6; sampling = false; sampler.join()
        val prefillMs = (first - t0) / 1e6; val decodeCps = text.length / ((System.nanoTime() - first) / 1e9)
        out.appendText("""{"bench":"memory","max_tokens":$tokens,"peak_pss_mb":${peak / 1024},"prefill_ms":$prefillMs,"total_ms":$totalMs,"chars":${text.length},"chars_per_s":$decodeCps}""" + "\n")
    }

    @Test fun greedyParity() = runBlocking {
        assumeTrue(c.models.scanAndVerify() != null)
        c.engineHolder.setMaxTokens(InstrumentationRegistry.getArguments().getString("tokens")?.toInt() ?: 2048)
        val ctx = InstrumentationRegistry.getInstrumentation().context
        val cases = SnJson.parseToJsonElement(ctx.assets.open("l7/cases.json").bufferedReader().readText()).jsonArray
        val pick = (InstrumentationRegistry.getArguments().getString("cases") ?: "0,1,2").split(",").map { it.toInt() }
        (c.llm as GemmaEngine).timeoutMs = 600_000L
        c.engineHolder.use { }
        var firstOut: String? = null
        for ((k, i) in (pick + pick.first()).withIndex()) {
            val cs = cases[i].jsonObject
            val tl = ArrayList<Pair<Long, Int>>(); val t0 = System.currentTimeMillis()
            val text = c.llm.generate(LlmTask.ANALYZE, cs["system"]!!.jsonPrimitive.content, cs["user"]!!.jsonPrimitive.content) { t ->
                tl += (System.currentTimeMillis() - t0) to t.length }
            val pc = cs["pc_output"]!!.jsonPrimitive.content
            // ETA quality: at the sample closest to half the final length, estimate vs the real time left
            val total = System.currentTimeMillis() - t0; val half = text.length / 2
            val est = EtaEstimator(expected = 1100); var etaAtHalf: Int? = null; var tHalf = 0L
            for ((t, ch) in tl) { if (ch == 0) continue; est.add(t, ch); if (etaAtHalf == null && ch >= half) { etaAtHalf = est.etaSec(); tHalf = t } }
            val repeatOf = if (k == pick.size) pick.first() else null
            if (k == 0) firstOut = text
            out.appendText(SnJson.encodeToString(kotlinx.serialization.json.JsonObject.serializer(), kotlinx.serialization.json.buildJsonObject {
                put("bench", kotlinx.serialization.json.JsonPrimitive("greedy")); put("case", kotlinx.serialization.json.JsonPrimitive(cs["id"]!!.jsonPrimitive.content))
                put("repeat", kotlinx.serialization.json.JsonPrimitive(repeatOf != null))
                put("identical_to_pc", kotlinx.serialization.json.JsonPrimitive(text.trim() == pc.trim()))
                put("identical_to_first_run", kotlinx.serialization.json.JsonPrimitive(if (repeatOf != null) text == firstOut else null))
                put("chars", kotlinx.serialization.json.JsonPrimitive(text.length)); put("total_ms", kotlinx.serialization.json.JsonPrimitive(total))
                put("eta_at_half_s", kotlinx.serialization.json.JsonPrimitive(etaAtHalf)); put("true_left_at_half_s", kotlinx.serialization.json.JsonPrimitive((total - tHalf) / 1000.0))
                put("out", kotlinx.serialization.json.JsonPrimitive(text)) }) + "\n")
        }
    }
}

private const val FAKE_ANALYSIS_V21 = """{"possible_categories":[{"key":"eczema_atopic","likelihood":"higher","why":"Supported by moderate itching on the arm."},{"key":"contact_dermatitis","likelihood":"possible","why":"Possible: a new product can look similar."},{"key":"psoriasis","likelihood":"less_likely","why":"Less likely: psoriasis plaques are thicker with silvery scale."}],"uncertainty":{"level":"low","reasons":[]},"explanation":"Based on your answers, it has been present for one to four weeks on the arm, with moderate itching. Eczema / atopic dermatitis fits best so far.","what_would_help":["A doctor's examination in person"],"self_care_info":["Moisturise often with a plain, fragrance-free cream."],"triage":{"tier":"LOW","advice":"Watch it and use gentle care."},"disagreement_with_image_model":false}"""
