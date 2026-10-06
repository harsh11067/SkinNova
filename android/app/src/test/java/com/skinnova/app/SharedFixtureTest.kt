package com.skinnova.app

import com.skinnova.app.ml.IntakeValidator
import com.skinnova.app.ml.OutputParser
import com.skinnova.app.ml.PromptBuilder
import com.skinnova.app.model.CvScore
import com.skinnova.app.model.Labels
import com.skinnova.app.model.QuestionnaireAnswers
import com.skinnova.app.model.SnJson
import com.skinnova.app.model.Tier
import com.skinnova.app.model.TimelineRuleInput
import com.skinnova.app.safety.ContentGuards
import com.skinnova.app.safety.RedFlagRules
import com.skinnova.app.safety.TierResolver
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.booleanOrNull
import kotlinx.serialization.json.decodeFromJsonElement
import kotlinx.serialization.json.doubleOrNull
import kotlinx.serialization.json.intOrNull
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import kotlin.random.Random

/**
 * Same JSON fixtures as ml/tests (pytest) — one oracle for both implementations (test.md S1–S3, V4, golden prompts).
 * Fixtures live in tests/fixtures (added as test resources); assets are read from src/main/assets.
 */
class SharedFixtureTest {
    private fun fixture(name: String) = SnJson.parseToJsonElement(javaClass.classLoader!!.getResource(name)!!.readText()).jsonArray
    private fun asset(path: String) = File("src/main/assets/$path").readText()
    private val labels = SnJson.decodeFromString(Labels.serializer(), asset("labels.json"))
    private val guards = ContentGuards(ContentGuards.parseTerms(asset("safety/rx_terms.txt")))
    private val topics = IntakeValidator.parseTopics(asset("safety/intake_topics.json"))
    private fun cv(e: JsonElement) = e.jsonArray.map { CvScore(it.jsonObject["key"]!!.jsonPrimitive.content, it.jsonObject["p"]!!.jsonPrimitive.doubleOrNull!!) }
    private fun answers(e: JsonElement) = SnJson.decodeFromJsonElement<QuestionnaireAnswers>(e)
    private fun strs(e: JsonElement?) = e!!.jsonArray.map { it.jsonPrimitive.content }

    @Test fun redflagCases() {
        val cases = fixture("redflag_cases.json")
        assertTrue(cases.size >= 60)
        for (c in cases) {
            val o = c.jsonObject
            val tl = o["timeline"].let { if (it == null || it is JsonNull) null else SnJson.decodeFromJsonElement<TimelineRuleInput>(it) }
            val r = RedFlagRules.evaluate(answers(o["answers"]!!), cv(o["cv"]!!), labels,
                o["quality_forced"]!!.jsonPrimitive.booleanOrNull!!, tl)
            val e = o["expected"]!!.jsonObject
            val id = o["id"]!!.jsonPrimitive.content
            assertEquals(id, e["tier"]!!.jsonPrimitive.content, r.tier.name)
            assertEquals(id, strs(e["fired"]).sorted(), r.fired.sorted())
            assertEquals(id, strs(e["messages"]).sorted(), r.messages.sorted())
            assertEquals(id, e["force_uncertainty_high"]!!.jsonPrimitive.booleanOrNull, r.forceUncertaintyHigh)
        }
    }

    @Test fun tierCases() {
        for (c in fixture("tier_cases.json")) {
            val o = c.jsonObject
            val floor = TierResolver.classFloor(cv(o["cv"]!!), labels)
            fun t(k: String) = (o[k] as? JsonPrimitive)?.takeIf { it.isString }?.let { Tier.parse(it.content) }
            val got = TierResolver.resolve(t("llm"), t("rule")!!, floor, t("timeline"))
            assertEquals(o["id"]!!.jsonPrimitive.content, o["expected"]!!.jsonPrimitive.content, got.name)
        }
    }

    @Test fun tierMonotoneProperty() {
        val rnd = Random(3407)
        val keys = labels.keys
        repeat(10_000) {
            val a = QuestionnaireAnswers(
                bodySite = com.skinnova.app.model.Enums.BODY_SITES.random(rnd), duration = com.skinnova.app.model.Enums.DURATIONS.random(rnd),
                itch = rnd.nextInt(4), pain = rnd.nextInt(4), changing = com.skinnova.app.model.Enums.CHANGING.random(rnd),
                bleedingOrCrusting = rnd.nextDouble() < .3, feverOrUnwell = rnd.nextDouble() < .2, othersAffected = rnd.nextDouble() < .2,
                newProductOrExposure = rnd.nextDouble() < .3, ageBand = com.skinnova.app.model.Enums.AGE_BANDS.random(rnd))
            val w = keys.map { rnd.nextDouble() }; val tot = w.sum()
            val scores = keys.zip(w).map { (k, x) -> CvScore(k, x / tot) }
            val r = RedFlagRules.evaluate(a, scores, labels)
            val floor = TierResolver.classFloor(scores, labels)
            val llm = (Tier.entries + listOf(null)).random(rnd)
            val fin = TierResolver.resolve(llm, r.tier, floor)
            assertTrue(fin.ordinal >= maxOf(r.tier.ordinal, floor.ordinal))
            if (llm != null) assertTrue(fin.ordinal >= llm.ordinal)
        }
    }

    @Test fun validatorCases() {
        val parser = OutputParser(labels.keys, guards)
        for (c in fixture("validator_cases.json")) {
            val o = c.jsonObject
            val id = o["id"]!!.jsonPrimitive.content
            val res = parser.parse(o["text"]!!.jsonPrimitive.content, (o["cv_top1_p"] as? JsonPrimitive)?.doubleOrNull,
                Tier.parse((o["rule_tier"] as? JsonPrimitive)?.content) ?: Tier.LOW)
            assertEquals("$id ${res.errors}", o["expect_ok"]!!.jsonPrimitive.booleanOrNull, res.ok)
            for (pre in strs(o["expect_error_prefixes"])) assertTrue("$id missing $pre in ${res.errors}", res.errors.any { it.startsWith(pre) })
            if (res.ok) {
                (o["expect_uncertainty"] as? JsonPrimitive)?.let { assertEquals(id, it.content, res.output!!.uncertainty.level) }
                (o["expect_tier"] as? JsonPrimitive)?.let { assertEquals(id, it.content, res.output!!.triage.tier) }
            }
        }
    }

    @Test fun intakeCases() {
        for (c in fixture("intake_cases.json")) {
            val o = c.jsonObject
            val id = o["id"]!!.jsonPrimitive.content
            val r = IntakeValidator.validate(o["text"]!!.jsonPrimitive.content, o["transcript"]!!.jsonPrimitive.content, topics)
            assertEquals(id, o["expect_ok"]!!.jsonPrimitive.booleanOrNull, r.ok)
            val kept = r.fields.filterValues { it != null }.mapValues { (_, v) -> v!!.value }
            val exp = o["expect_fields"]!!.jsonObject
            assertEquals(id, exp.keys, kept.keys)
            exp.forEach { (k, v) -> assertEquals("$id/$k", norm(v), norm(kept[k]!!)) }
            assertEquals(id, o["expect_dropped"]!!.jsonObject.mapValues { it.value.jsonPrimitive.content }, r.dropped)
            (o["expect_notes_len"] as? JsonPrimitive)?.intOrNull?.let { assertEquals(id, it, r.unparsedNotes.length) }
        }
    }

    private fun norm(e: JsonElement): String = (e as JsonPrimitive).let { it.booleanOrNull?.toString() ?: it.intOrNull?.toString() ?: it.content }

    @Test fun promptGoldenByteIdentical() {
        val pb = PromptBuilder { File("src/main/assets/$it").readText() }
        for (c in fixture("prompt_golden.json")) {
            val o = c.jsonObject
            val id = o["id"]!!.jsonPrimitive.content
            if (id == "repair") {
                assertEquals(o["expected_repair"]!!.jsonPrimitive.content,
                    pb.repair(o["previous"]!!.jsonPrimitive.content, strs(o["errors"])))
                continue
            }
            val p = pb.analysis(answers(o["answers"]!!), cv(o["cv"]!!), o["rule_tier"]!!.jsonPrimitive.content, strs(o["rule_messages"]))
            assertEquals(id, o["expected_system"]!!.jsonPrimitive.content, p.system)
            assertEquals(id, o["expected_user"]!!.jsonPrimitive.content, p.user)
            assertEquals(id, o["version"]!!.jsonPrimitive.content, p.version)
        }
    }

}
