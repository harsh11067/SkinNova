package com.skinnova.app.ml

import com.skinnova.app.model.AnalysisOutput
import com.skinnova.app.model.Enums
import com.skinnova.app.model.SnJson
import com.skinnova.app.model.Tier
import com.skinnova.app.safety.ContentGuards
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.booleanOrNull
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.put

/** contracts §3 validator rules 1–7. Python twin: ml/llm/validate.py (shared fixture validator_cases.json). */
data class ParseResult(val ok: Boolean, val output: AnalysisOutput? = null, val errors: List<String> = emptyList())

class OutputParser(private val labelKeys: List<String>, private val guards: ContentGuards) {
    companion object {
        private val SENTENCE = Regex("""[^.!?]+[.!?]+|[^.!?]+$""")

        /** Rule 1: first balanced top-level JSON object; truncated output → null (never an inner object). */
        fun extractJson(text: String): String? {
            val t = text.replace("```json", "```")
            val start = t.indexOf('{')
            if (start < 0) return null
            var depth = 0; var inStr = false; var esc = false
            for (i in start until t.length) {
                val ch = t[i]
                if (inStr) {
                    if (esc) esc = false else if (ch == '\\') esc = true else if (ch == '"') inStr = false
                } else when (ch) {
                    '"' -> inStr = true
                    '{' -> depth++
                    '}' -> { depth--; if (depth == 0) return t.substring(start, i + 1) }
                }
            }
            return null
        }

        private val DECIMAL = Regex("""(\d)\.(\d)""")

        /** Decimals ("8.9 units") are not sentence ends — mirrors ml/llm/validate.py count_sentences. */
        fun countSentences(text: String) = SENTENCE.findAll(DECIMAL.replace(text.trim(), "$1,$2")).count { it.value.isNotBlank() }
    }

    private fun JsonElement?.str(): String? = (this as? JsonPrimitive)?.takeIf { it.isString }?.content
    private fun JsonElement?.strList(): List<String>? =
        (this as? JsonArray)?.let { arr -> if (arr.all { it is JsonPrimitive && it.isString }) arr.map { it.jsonPrimitive.content } else null }

    fun parse(text: String, cvTop1P: Double? = null, ruleTier: Tier = Tier.LOW): ParseResult {
        val raw = extractJson(text) ?: return ParseResult(false, errors = listOf("no_json_object"))
        val d = try { SnJson.parseToJsonElement(raw) } catch (e: Exception) {
            return ParseResult(false, errors = listOf("json_parse:${e.message?.take(60)}"))
        } as? JsonObject ?: return ParseResult(false, errors = listOf("not_object"))
        val errs = mutableListOf<String>()

        val catsEl = d["possible_categories"]
        val cats = (catsEl as? JsonArray) ?: JsonArray(emptyList())
        if (catsEl !is JsonArray || cats.size !in 1..3) errs += "categories_count"
        val seen = mutableSetOf<String?>()
        for (c in cats) {
            if (c !is JsonObject) { errs += "category_not_object"; continue }
            val k = c["key"].str()
            if (k !in labelKeys) errs += "unknown_key:$k"
            if (k in seen) errs += "duplicate_key:$k"
            seen += k
            val lk = c["likelihood"].str()
            if (lk !in Enums.LIKELIHOODS) errs += "bad_likelihood:$lk"
            val why = c["why"].str()
            if (why == null || why.isBlank()) errs += "missing_why" else if (why.length > 200) errs += "why_too_long"
        }
        val unc = d["uncertainty"] as? JsonObject
        if (unc == null || unc["level"].str() !in Enums.UNCERTAINTY) errs += "bad_uncertainty"
        else if (unc["reasons"] != null && unc["reasons"] !is JsonArray) errs += "bad_uncertainty_reasons"
        val tri = d["triage"] as? JsonObject
        if (tri == null || Tier.parse(tri["tier"].str()) == null || tri["advice"].str() == null) errs += "bad_triage"
        val exp = d["explanation"].str()
        if (exp == null || exp.isBlank()) errs += "missing_explanation"
        else { val n = countSentences(exp); if (n !in 1..6) errs += "explanation_sentences:$n" }
        for (key in listOf("what_would_help", "self_care_info")) {
            val v = d[key]
            if (v != null && v.strList() == null) errs += "bad_$key"
        }
        val dis = d["disagreement_with_image_model"]
        if (dis != null && (dis !is JsonPrimitive || dis.isString || dis.booleanOrNull == null)) errs += "bad_disagreement"

        val guarded = buildList {
            add(exp ?: "")
            addAll(d["self_care_info"].strList() ?: emptyList())
            cats.forEach { c -> (c as? JsonObject)?.get("why").str()?.let { add(it) } }
        }
        guarded.forEach { errs += guards.check(it) }
        if (errs.isNotEmpty()) return ParseResult(false, errors = errs)

        // Normalisations (rule 3 upgrade, rule 4 tier floor)
        val disagree = (dis as? JsonPrimitive)?.booleanOrNull ?: false
        var level = unc!!["level"].str()!!
        if (level == "low" && ((cvTop1P != null && cvTop1P < 0.5) || disagree)) level = "moderate"
        val tier = Tier.max(Tier.parse(tri!!["tier"].str())!!, ruleTier)
        val norm = buildJsonObject {
            d.forEach { (k, v) -> if (k != "uncertainty" && k != "triage") put(k, v) }
            put("uncertainty", JsonObject(unc.toMutableMap().apply { put("level", JsonPrimitive(level)) }))
            put("triage", JsonObject(tri.toMutableMap().apply { put("tier", JsonPrimitive(tier.name)) }))
            if (!d.containsKey("disagreement_with_image_model")) put("disagreement_with_image_model", false)
        }
        val out = try { SnJson.decodeFromJsonElement(AnalysisOutput.serializer(), norm) } catch (e: Exception) {
            return ParseResult(false, errors = listOf("decode:${e.message?.take(60)}"))
        }
        return ParseResult(true, out)
    }
}

/** Narration / translation guard: plain text, sentence bounds, content guards (contracts §8, §10). */
fun validateNarration(text: String, guards: ContentGuards, mustMentionLow: Boolean): List<String> {
    val errs = mutableListOf<String>()
    val t = text.trim()
    val n = OutputParser.countSentences(t)
    if (n !in 2..4) errs += "narration_sentences:$n"
    errs += guards.check(t)
    if (mustMentionLow && !Regex("uncertain|low confidence|not sure|can't be sure|cannot be sure", RegexOption.IGNORE_CASE).containsMatchIn(t))
        errs += "low_confidence_not_stated"
    return errs
}

