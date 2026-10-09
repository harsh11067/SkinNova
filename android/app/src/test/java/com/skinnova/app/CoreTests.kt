package com.skinnova.app

import com.skinnova.app.ml.AnalysisPipeline
import com.skinnova.app.ml.AnalysisState
import com.skinnova.app.ml.Fallback
import com.skinnova.app.ml.ImageOps
import com.skinnova.app.ml.Llm
import com.skinnova.app.ml.LlmTask
import com.skinnova.app.ml.OutputParser
import com.skinnova.app.ml.PromptBuilder
import com.skinnova.app.ml.Rgb
import com.skinnova.app.ml.Wav
import com.skinnova.app.model.CvScore
import com.skinnova.app.model.Labels
import com.skinnova.app.model.Mode
import com.skinnova.app.model.QuestionnaireAnswers
import com.skinnova.app.model.SnJson
import com.skinnova.app.safety.ContentGuards
import com.skinnova.app.timeline.Colour
import kotlinx.coroutines.test.runTest
import kotlinx.serialization.json.doubleOrNull
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assert.assertNull
import org.junit.Assert.assertFalse
import org.junit.Test
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs
import kotlin.math.sin

class CoreTests {
    private fun asset(p: String) = File("src/main/assets/$p").readText()
    private val labels = SnJson.decodeFromString(Labels.serializer(), asset("labels.json"))
    private val guards = ContentGuards(ContentGuards.parseTerms(asset("safety/rx_terms.txt")))
    private val prompts = PromptBuilder { asset(it) }
    private val parser = OutputParser(labels.keys, guards)

    // ---------- timeline colour (TL3 building block) ----------
    @Test fun ciede2000_sharma34() {
        val d = SnJson.parseToJsonElement(javaClass.classLoader!!.getResource("ciede2000_cases.json")!!.readText()).jsonObject
        for (p in d["pairs"]!!.jsonArray) {
            val v = p.jsonArray.map { it.jsonPrimitive.doubleOrNull!! }
            val got = Colour.ciede2000(doubleArrayOf(v[0], v[1], v[2]), doubleArrayOf(v[3], v[4], v[5]))
            assertEquals("pair $v", v[6], got, 1e-4)
        }
    }

    @Test fun jsDivergenceBounds() {
        assertEquals(0.0, Colour.jsDivergence(doubleArrayOf(.5, .5), doubleArrayOf(.5, .5)), 1e-9)
        assertEquals(1.0, Colour.jsDivergence(doubleArrayOf(1.0, 0.0), doubleArrayOf(0.0, 1.0)), 1e-6)
    }

    // ---------- V1 audio ----------
    @Test fun wavHeaderAndVad() {
        val rate = Wav.RATE
        val pcm = ShortArray(rate * 3) { i -> if (i in rate until 2 * rate) (8000 * sin(i * 2 * Math.PI * 440 / rate)).toInt().toShort() else 0 }
        val trimmed = Wav.trimSilence(pcm)
        assertTrue("silence trimmed", trimmed.size < pcm.size && trimmed.size >= rate)
        val wav = Wav.encode(trimmed)
        val b = ByteBuffer.wrap(wav).order(ByteOrder.LITTLE_ENDIAN)
        assertEquals("RIFF", String(wav, 0, 4)); assertEquals("WAVE", String(wav, 8, 4)); assertEquals("data", String(wav, 36, 4))
        assertEquals(1, b.getShort(22).toInt()); assertEquals(16000, b.getInt(24)); assertEquals(16, b.getShort(34).toInt())
        assertEquals(trimmed.size * 2, b.getInt(40))
        assertEquals(rate * Wav.MAX_SECONDS, Wav.cap(ShortArray(rate * 40)).size)
        assertEquals(0, Wav.trimSilence(ShortArray(rate)).size)   // pure silence → nothing to transcribe
    }

    // ---------- C4 preprocessing building block: PIL resize port vs Python golden ----------
    @Test fun pilResizeMatchesPython() {
        val g = SnJson.parseToJsonElement(javaClass.classLoader!!.getResource("pil_resize_golden.json")!!.readText()).jsonObject
        for (case in g["cases"]!!.jsonArray) {
            val o = case.jsonObject
            val w = o["w"]!!.jsonPrimitive.int; val h = o["h"]!!.jsonPrimitive.int
            val src = Rgb(w, h, o["src"]!!.jsonArray.map { it.jsonPrimitive.int }.toIntArray())
            val ow = o["ow"]!!.jsonPrimitive.int; val oh = o["oh"]!!.jsonPrimitive.int
            val f = if (o["filter"]!!.jsonPrimitive.content == "bilinear") ImageOps.Filter.BILINEAR else ImageOps.Filter.LANCZOS
            val got = ImageOps.resize(src, ow, oh, f).px
            val exp = o["out"]!!.jsonArray.map { it.jsonPrimitive.int }.toIntArray()
            var maxd = 0
            for (i in exp.indices) for (s in intArrayOf(16, 8, 0)) maxd = maxOf(maxd, abs(((got[i] shr s) and 255) - ((exp[i] shr s) and 255)))
            assertTrue("${o["name"]} max channel diff $maxd", maxd <= 1)
        }
    }

    // ---------- A1 / S7: pipeline with a fake LLM (full + repair + Basic-mode fallback) ----------
    private val answers = QuestionnaireAnswers("hand", "1_4w", 3, 0, "spreading", false, false, true, false, "18_39")
    private val cv = listOf(CvScore("tinea", .62), CvScore("eczema_atopic", .2), CvScore("scabies", .1), CvScore("other", .08))
    private val valid = """{"possible_categories":[{"key":"tinea","likelihood":"higher","why":"Itching and spreading fit."},{"key":"scabies","likelihood":"possible","why":"Others at home itch."}],"uncertainty":{"level":"low","reasons":[]},"explanation":"You said it itches a lot and others at home have it. A ring that spreads often fits a fungal infection.","what_would_help":["A clearer photo"],"self_care_info":["Keep the area clean and dry."],"triage":{"tier":"LOW","advice":"x"},"disagreement_with_image_model":false}"""

    private class FakeLlm(val replies: MutableList<String>, val throwOn: Int = -1) : Llm {
        override val available = true
        var calls = 0
        override suspend fun generate(task: LlmTask, system: String, user: String, imagePath: String?, audio: ByteArray?, onToken: (String) -> Unit): String {
            if (calls++ == throwOn) error("engine crashed")
            return replies.removeAt(0).also { onToken(it) }
        }
    }

    private fun pipeline(llm: Llm) = AnalysisPipeline(labels, prompts, parser, guards, llm)

    @Test fun fullModeAndTierNeverLowered() = runTest {
        val states = mutableListOf<AnalysisState>()
        val r = pipeline(FakeLlm(mutableListOf(valid))).run(answers, cv, false, null, "en") { states += it }
        assertEquals(Mode.full, r.mode)
        assertEquals("MODERATE", r.finalTier)              // LLM said LOW; R7 (others + itch 3) → MODERATE wins
        assertTrue("rf_r7" in r.ruleMessages)
        assertTrue(states.any { it is AnalysisState.Generating })
    }

    /** v2.1 item 2: a complete early result comes first (before any generation), and the final tier is never lower. */
    @Test fun earlyResultFirstAndTierNeverDrops() = runTest {
        val states = mutableListOf<AnalysisState>()
        val r = pipeline(FakeLlm(mutableListOf(valid))).run(answers, cv, false, null, "en") { states += it }
        val iPre = states.indexOfFirst { it is AnalysisState.Preliminary }; val iGen = states.indexOfFirst { it is AnalysisState.Generating }
        assertTrue("Preliminary before Generating", iPre in 0 until iGen)
        val pre = (states[iPre] as AnalysisState.Preliminary).result
        assertEquals(Mode.basic, pre.mode); assertEquals(AnalysisPipeline.PENDING, pre.fallbackReason)
        assertTrue(pre.output.possibleCategories.isNotEmpty() && pre.output.explanation.isNotBlank())
        assertFalse("no 'Basic mode' reason while pending", pre.output.uncertainty.reasons.any { it.startsWith("Basic mode") })
        assertTrue(com.skinnova.app.model.Tier.parse(r.finalTier)!! >= com.skinnova.app.model.Tier.parse(pre.finalTier)!!)
        assertTrue(parser.parse(SnJson.encodeToString(com.skinnova.app.model.AnalysisOutput.serializer(), pre.output)).ok)
    }

    @Test fun noEarlyResultWithoutModel() = runTest {
        val states = mutableListOf<AnalysisState>()
        val noLlm = object : Llm { override val available = false
            override suspend fun generate(task: LlmTask, system: String, user: String, imagePath: String?, audio: ByteArray?, onToken: (String) -> Unit) = error("no") }
        pipeline(noLlm).run(answers, cv, false, null, "en") { states += it }
        assertTrue(states.none { it is AnalysisState.Preliminary })
    }

    /** v2.1 item 3: time left from the last ~5 s of writing. */
    @Test fun etaFromRecentWritingSpeed() {
        val e = com.skinnova.app.ui.EtaEstimator(expected = 1100)
        assertNull(e.etaSec())
        e.add(0, 0); e.add(1_000, 20); e.add(2_000, 40)              // 20 chars/s
        assertEquals(53, e.etaSec())                                    // (1100 − 40) / 20
        e.add(9_000, 180); e.add(10_000, 200)                           // window keeps the last 5 s: 20 chars/s
        assertEquals(45, e.etaSec())
        assertEquals(0.97f, e.fraction(5000))
    }

    @Test fun repairThenSuccess() = runTest {
        val llm = FakeLlm(mutableListOf("Sure! here is: {\"possible_categories\": [", valid))
        val r = pipeline(llm).run(answers, cv, false, null, "en") {}
        assertEquals(Mode.full, r.mode); assertEquals(2, llm.calls)
    }

    @Test fun doubleInvalidFallsBackToBasic() = runTest {
        val bad = valid.replace("Keep the area clean and dry.", "Apply clotrimazole 1% cream twice daily.")
        val r = pipeline(FakeLlm(mutableListOf(bad, bad))).run(answers, cv, false, null, "en") {}
        assertEquals(Mode.basic, r.mode); assertEquals("INVALID_JSON", r.fallbackReason)
        assertTrue(r.ruleMessages.contains("rf_r7"))
        assertTrue(parser.parse(SnJson.encodeToString(com.skinnova.app.model.AnalysisOutput.serializer(), r.output)).ok)  // basic output itself is valid
    }

    @Test fun engineErrorFallsBack() = runTest {
        val r = pipeline(FakeLlm(mutableListOf(), throwOn = 0)).run(answers, cv, true, null, "en") {}
        assertEquals(Mode.basic, r.mode); assertEquals("ENGINE_ERROR", r.fallbackReason)
        assertEquals("high", r.output.uncertainty.level)   // R8 forced
    }

    @Test fun basicModeHedges() {
        val o = Fallback.build(answers, listOf(CvScore("acne", .41), CvScore("other", .3), CvScore("tinea", .29)), prompts.cards, false,
            com.skinnova.app.model.Tier.MODERATE)
        assertEquals("moderate", o.uncertainty.level)
        assertEquals("possible", o.possibleCategories.first().likelihood)
        assertTrue(guards.check(o.explanation).isEmpty())
    }

}
