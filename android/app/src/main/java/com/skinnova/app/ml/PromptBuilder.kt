package com.skinnova.app.ml

import com.skinnova.app.model.CvScore
import com.skinnova.app.model.QuestionnaireAnswers
import com.skinnova.app.model.SnJson
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.put
import java.math.BigDecimal
import java.math.RoundingMode

/**
 * Fills assets/prompts/ templates. Must render byte-identically to ml/llm/prompt_builder.py, which renders the SFT
 * training data (CLAUDE.md non-negotiable 7). Golden files: tests/fixtures/prompt_golden.json.
 */
data class Prompt(val system: String, val user: String, val version: String)

class PromptBuilder(private val readAsset: (String) -> String) {
    companion object {
        const val SCHEMA_COMPACT = "{\"possible_categories\":[{\"key\":\"<label key>\",\"likelihood\":\"higher|possible|less_likely\",\"why\":\"<=200 chars\"}] (1-3)," +
            "\"uncertainty\":{\"level\":\"low|moderate|high\",\"reasons\":[\"...\"]},\"explanation\":\"2-5 sentences citing an answer\"," +
            "\"what_would_help\":[\"...\"],\"self_care_info\":[\"general, non-prescription\"]," +
            "\"triage\":{\"tier\":\"LOW|MODERATE|HIGH|URGENT\",\"advice\":\"...\"},\"disagreement_with_image_model\":false}"
        val CARD_FIELDS = listOf("summary", "typical_features", "distinguishing_from", "see_doctor_if")

        /** Python round(x, 2) on a float = exact binary value, half-even. repr → shortest string. */
        fun round2(x: Double): Double = BigDecimal(x).setScale(2, RoundingMode.HALF_EVEN).toDouble()

        fun sanitizeFreeText(s: String) = s.replace("<<<", "").replace(">>>", "").take(200)

        fun answersJson(a: QuestionnaireAnswers): JsonObject = buildJsonObject {
            put("body_site", a.bodySite); put("duration", a.duration); put("itch", a.itch); put("pain", a.pain)
            put("changing", a.changing); put("bleeding_or_crusting", a.bleedingOrCrusting); put("fever_or_unwell", a.feverOrUnwell)
            put("others_affected", a.othersAffected); put("new_product_or_exposure", a.newProductOrExposure)
            put("age_band", a.ageBand); put("skin_tone", a.skinTone)
        }

        fun top3(cv: List<CvScore>) = cv.sortedByDescending { it.p }.take(3).map { CvScore(it.key, round2(it.p)) }

        fun cj(e: JsonElement): String = SnJson.encodeToString(JsonElement.serializer(), e)
    }

    private val templates = HashMap<String, Pair<String, String>>()
    val cards: JsonObject by lazy { SnJson.parseToJsonElement(readAsset("prompts/condition_cards.json")).jsonObject["cards"]!!.jsonObject }

    fun template(name: String): Pair<String, String> = templates.getOrPut(name) {
        val text = readAsset("prompts/$name")
        val nl = text.indexOf('\n')
        val first = text.substring(0, nl)
        require(first.startsWith("# v")) { "$name missing version header" }
        first.substring(2).trim() to text.substring(nl + 1).trimEnd('\n')
    }

    fun analysis(answers: QuestionnaireAnswers, cv: List<CvScore>, ruleTier: String, ruleMessages: List<String>): Prompt {
        val (vs, system) = template("analyze_system.txt")
        val (vu, userT) = template("analyze_user.txt")
        check(vs == vu)
        val t3 = top3(cv)
        val t3Json = buildJsonArray { t3.forEach { s -> add(buildJsonObject { put("key", s.key); put("p", s.p) }) } }
        val notes = buildJsonObject {
            t3.forEach { s ->
                val c = cards[s.key] as? JsonObject ?: return@forEach
                put(s.key, buildJsonObject { CARD_FIELDS.forEach { f -> c[f]?.let { put(f, it) } } })
            }
        }
        val user = userT.replace("{cv_top3_json}", cj(t3Json))
            .replace("{rule_tier}", ruleTier)
            .replace("{rule_messages_json}", cj(JsonArray(ruleMessages.map { JsonPrimitive(it) })))
            .replace("{answers_json}", cj(answersJson(answers)))
            .replace("{free_text}", sanitizeFreeText(answers.freeText))
            .replace("{cards_json}", cj(notes))
            .replace("{schema_compact}", SCHEMA_COMPACT)
        return Prompt(system, user, vs)
    }

    fun repair(previous: String, errors: List<String>): String =
        template("repair.txt").second.replace("{errors}", errors.joinToString("; ")).replace("{schema_compact}", SCHEMA_COMPACT)
            .replace("{previous}", previous.replace("<<<", "").replace(">>>", ""))

    /** Ask SkinNova follow-up (prompts/chat_*.txt; probe: ml/eval/chat_probe.py renders the same way). */
    fun chat(r: com.skinnova.app.model.FinalResult, earlier: List<Pair<String, String>>, question: String, lang: String,
             careHome: List<String> = emptyList(), careFood: List<String> = emptyList()): Prompt {
        val (vs, system) = template("chat_system.txt")
        val (_, userT) = template("chat_user.txt")
        val a = r.answers
        val result = buildJsonObject {
            put("possible_categories", buildJsonArray { r.output.possibleCategories.forEach { c -> add(buildJsonObject { put("key", c.key); put("likelihood", c.likelihood) }) } })
            put("explanation", r.output.explanation)
            put("answers", buildJsonObject { put("body_site", a.bodySite); put("duration", a.duration); put("itch", a.itch); put("pain", a.pain)
                put("changing", a.changing); put("age_band", a.ageBand) })
        }
        val notes = buildJsonObject {
            r.output.possibleCategories.forEach { c ->
                val card = cards[c.key] as? JsonObject ?: return@forEach
                put(c.key, buildJsonObject { CARD_FIELDS.forEach { f -> card[f]?.let { put(f, it) } } })
            }
        }
        val prior = buildJsonArray { earlier.takeLast(2).forEach { (q, ans) -> add(buildJsonObject { put("q", q); put("a", ans) }) } }
        // reviewed home-care + food notes of the top category (assets/care/relief.json, English) — never the pharmacy list
        val care = buildJsonObject {
            put("home", buildJsonArray { careHome.forEach { add(JsonPrimitive(it)) } }); put("food", buildJsonArray { careFood.forEach { add(JsonPrimitive(it)) } })
        }
        val user = userT.replace("{result_json}", cj(result)).replace("{advice_level}", r.finalTier).replace("{notes_json}", cj(notes))
            .replace("{care_json}", cj(care))
            .replace("{earlier_json}", cj(prior)).replace("{language}", if (lang == "hi") "Hindi" else "English")
            .replace("{question}", question.replace("<<<", "").replace(">>>", "").take(300))
        return Prompt(system, user, vs)
    }

    fun transcribe(langHint: String) = template("transcribe.txt").second.replace("{lang_hint}", langHint)
    fun extractSystem() = template("extract_system.txt").second
    fun narrateSystem() = template("narrate_system.txt").second
    fun translateSystem(targetLanguage: String) = template("translate_system.txt").second.replace("{target_language}", targetLanguage)
    val version: String get() = template("analyze_system.txt").first
}
