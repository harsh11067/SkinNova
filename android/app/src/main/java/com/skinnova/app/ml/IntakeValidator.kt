package com.skinnova.app.ml

import com.skinnova.app.model.Enums
import com.skinnova.app.model.SnJson
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.booleanOrNull
import kotlinx.serialization.json.intOrNull
import java.text.Normalizer

/**
 * contracts §9 — Python twin ml/voice/intake.py (shared fixture intake_cases.json).
 * Voice intake never guesses: a field survives only with a valid value AND an evidence quote found in the transcript AND,
 * for the yes/no and 0–3 fields, a quote that mentions the field's topic (assets/safety/intake_topics.json: a real quote
 * attached to the wrong field — "nobody else has it" → bleeding = false — is dropped).
 */
data class IntakeField(val value: JsonElement, val evidence: String)

data class IntakeResult(
    val ok: Boolean,
    val language: String?,
    val fields: Map<String, IntakeField?>,
    val dropped: Map<String, String>,
    val unparsedNotes: String,
)

object IntakeValidator {
    private enum class Kind { ENUM, SCALE, BOOL }
    private val FIELDS: Map<String, Pair<Kind, List<String>?>> = linkedMapOf(
        "body_site" to (Kind.ENUM to Enums.BODY_SITES), "duration" to (Kind.ENUM to Enums.DURATIONS),
        "changing" to (Kind.ENUM to Enums.CHANGING), "age_band" to (Kind.ENUM to Enums.AGE_BANDS),
        "itch" to (Kind.SCALE to null), "pain" to (Kind.SCALE to null),
        "bleeding_or_crusting" to (Kind.BOOL to null), "fever_or_unwell" to (Kind.BOOL to null),
        "others_affected" to (Kind.BOOL to null), "new_product_or_exposure" to (Kind.BOOL to null),
    )
    val FIELD_NAMES: Set<String> get() = FIELDS.keys
    private const val MIN_SIM = 0.8

    fun norm(s: String) = Normalizer.normalize(s, Normalizer.Form.NFC).lowercase().replace(Regex("\\s+"), " ").trim()

    fun lev(a: String, b: String): Int {
        var prev = IntArray(b.length + 1) { it }
        for (i in 1..a.length) {
            val cur = IntArray(b.length + 1); cur[0] = i
            for (j in 1..b.length) cur[j] = minOf(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + if (a[i - 1] == b[j - 1]) 0 else 1)
            prev = cur
        }
        return prev[b.length]
    }

    fun evidenceOk(evidence: String, transcript: String): Boolean {
        val e = norm(evidence); val t = norm(transcript)
        if (e.length < 2) return false
        if (t.contains(e)) return true
        val n = e.length
        if (n > t.length) return 1.0 - lev(e, t).toDouble() / maxOf(n, t.length) >= MIN_SIM
        var best = 0.0
        for (i in 0..t.length - n) best = maxOf(best, 1.0 - lev(e, t.substring(i, i + n)).toDouble() / n)
        return best >= MIN_SIM
    }

    /** assets/safety/intake_topics.json → field → terms ("_doc" keys skipped). */
    fun parseTopics(json: String): Map<String, List<String>> =
        (SnJson.parseToJsonElement(json) as JsonObject).filterKeys { !it.startsWith("_") }
            .mapValues { (_, v) -> (v as JsonArray).map { (it as JsonPrimitive).content } }

    /** ml/voice/intake.py topic_ok: Latin-script terms must start a word ("ill" ≠ "will"); Devanagari terms match anywhere. */
    fun topicOk(field: String, evidence: String, topics: Map<String, List<String>>): Boolean {
        val terms = topics[field] ?: return true
        val e = norm(evidence)
        return terms.map(::norm).any { t ->
            if (t.all { it.code < 128 }) Regex("(?<![a-z])" + Regex.escape(t)).containsMatchIn(e) else e.contains(t)
        }
    }

    private fun valueOk(kind: Kind, allowed: List<String>?, v: JsonElement): Boolean {
        val p = v as? JsonPrimitive ?: return false
        return when (kind) {
            Kind.ENUM -> p.isString && p.content in allowed!!
            Kind.SCALE -> !p.isString && p.booleanOrNull == null && p.content.toIntOrNull()?.let { it in 0..3 } == true
            Kind.BOOL -> !p.isString && p.booleanOrNull != null
        }
    }

    fun validate(text: String, transcript: String, topics: Map<String, List<String>>): IntakeResult {
        val empty = FIELDS.keys.associateWith { null as IntakeField? }
        fun fail(r: String) = IntakeResult(false, null, empty, mapOf("_all" to r), "")
        val raw = OutputParser.extractJson(text) ?: return fail("no_json_object")
        val d = try { SnJson.parseToJsonElement(raw) } catch (e: Exception) { return fail("json_parse") }
        if (d !is JsonObject || d["fields"] !is JsonObject) return fail("bad_shape")
        val fields = empty.toMutableMap()
        val dropped = linkedMapOf<String, String>()
        for ((f, item) in d["fields"] as JsonObject) {
            val spec = FIELDS[f]
            if (spec == null) { dropped[f] = "unknown_field"; continue }
            if (item is JsonNull) continue
            val obj = item as? JsonObject
            val ev = (obj?.get("evidence") as? JsonPrimitive)?.takeIf { it.isString }?.content
            if (obj == null || !obj.containsKey("value") || ev == null) { dropped[f] = "bad_item"; continue }
            if (!valueOk(spec.first, spec.second, obj["value"]!!)) { dropped[f] = "bad_value"; continue }
            if (!evidenceOk(ev, transcript)) { dropped[f] = "evidence_not_in_transcript"; continue }
            if (!topicOk(f, ev, topics)) { dropped[f] = "evidence_off_topic"; continue }
            fields[f] = IntakeField(obj["value"]!!, ev)
        }
        val lang = (d["language"] as? JsonPrimitive)?.takeIf { it.isString }?.content
        val notes = (d["unparsed_notes"] as? JsonPrimitive)?.takeIf { it.isString }?.content?.take(200) ?: ""
        return IntakeResult(true, lang, fields, dropped, notes)
    }

    fun intValue(f: IntakeField?) = (f?.value as? JsonPrimitive)?.intOrNull
    fun boolValue(f: IntakeField?) = (f?.value as? JsonPrimitive)?.booleanOrNull
    fun strValue(f: IntakeField?) = (f?.value as? JsonPrimitive)?.takeIf { it.isString }?.content
}
