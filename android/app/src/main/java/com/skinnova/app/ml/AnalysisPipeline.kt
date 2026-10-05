package com.skinnova.app.ml

import com.skinnova.app.model.AnalysisOutput
import com.skinnova.app.model.Category
import com.skinnova.app.model.CvScore
import com.skinnova.app.model.FinalResult
import com.skinnova.app.model.Labels
import com.skinnova.app.model.Localized
import com.skinnova.app.model.Mode
import com.skinnova.app.model.QuestionnaireAnswers
import com.skinnova.app.model.SnJson
import com.skinnova.app.model.Tier
import com.skinnova.app.model.Triage
import com.skinnova.app.model.Uncertainty
import com.skinnova.app.safety.ContentGuards
import com.skinnova.app.safety.RedFlagRules
import com.skinnova.app.safety.TierResolver
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.TimeoutCancellationException
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.put

/** architecture §3 state machine. */
sealed interface AnalysisState {
    data object Idle : AnalysisState
    data object CheckingPhoto : AnalysisState
    data object Classifying : AnalysisState
    data object Rules : AnalysisState
    data object LoadingModel : AnalysisState
    data class Generating(val partialText: String) : AnalysisState
    data object Validating : AnalysisState
    data object Translating : AnalysisState
    data class Done(val result: FinalResult) : AnalysisState
    data class Failed(val message: String) : AnalysisState
}

enum class FallbackReason { NO_MODEL, ENGINE_ERROR, INVALID_JSON, TIMEOUT }

/**
 * Pure orchestration (no Android types) so the full path incl. Basic mode is JVM-testable with a fake [Llm] (test.md A1, S7).
 */
class AnalysisPipeline(
    private val labels: Labels,
    private val prompts: PromptBuilder,
    private val parser: OutputParser,
    private val guards: ContentGuards,
    private val llm: Llm,
    private val isModelLoaded: () -> Boolean = { true },
    private val modelSha: () -> String = { "" },
) {
    suspend fun run(
        answers: QuestionnaireAnswers, cv: List<CvScore>, qualityForced: Boolean, imagePath: String?, lang: String,
        emit: (AnalysisState) -> Unit,
    ): FinalResult {
        emit(AnalysisState.Rules)
        val rules = RedFlagRules.evaluate(answers, cv, labels, qualityForced)
        val floor = TierResolver.classFloor(cv, labels)
        val top3 = RedFlagRules.top3(cv)
        val top1p = top3.firstOrNull()?.p

        var output: AnalysisOutput? = null
        var reason: FallbackReason? = null
        if (!llm.available) reason = FallbackReason.NO_MODEL else {
            try {
                if (!isModelLoaded()) emit(AnalysisState.LoadingModel)
                val p = prompts.analysis(answers, cv, rules.tier.name, rules.messages)
                val first = llm.generate(LlmTask.ANALYZE, p.system, p.user, imagePath) { emit(AnalysisState.Generating(it)) }
                emit(AnalysisState.Validating)
                var res = parser.parse(first, top1p, rules.tier)
                if (!res.ok) {
                    val again = llm.generate(LlmTask.REPAIR, p.system, prompts.repair(first, res.errors), null) { emit(AnalysisState.Generating(it)) }
                    emit(AnalysisState.Validating)
                    res = parser.parse(again, top1p, rules.tier)
                }
                if (res.ok) output = res.output else reason = FallbackReason.INVALID_JSON
            } catch (e: TimeoutCancellationException) {
                reason = FallbackReason.TIMEOUT
            } catch (e: CancellationException) {
                throw e
            } catch (e: Throwable) {
                reason = FallbackReason.ENGINE_ERROR
            }
        }
        val llmTier = output?.let { Tier.parse(it.triage.tier) }
        val finalTier = TierResolver.resolve(llmTier, rules.tier, floor)
        var out = output ?: Fallback.build(answers, top3, prompts.cards, rules.forceUncertaintyHigh, finalTier)
        if (rules.forceUncertaintyHigh && out.uncertainty.level != "high")
            out = out.copy(uncertainty = out.uncertainty.copy(level = "high"))
        out = out.copy(triage = out.triage.copy(tier = finalTier.name))

        var localized: Localized? = null
        if (lang != "en" && output != null && llm.available) {
            emit(AnalysisState.Translating)
            localized = runCatching { translate(out, lang) }.getOrNull()
        }
        return FinalResult(
            output = out, cvTop3 = top3, ruleMessages = rules.messages, rulesFired = rules.fired, finalTier = finalTier.name,
            mode = if (output != null) Mode.full else Mode.basic, fallbackReason = reason?.name, promptVersion = prompts.version,
            modelSha = modelSha(), lang = lang, localized = localized, answers = answers, qualityForced = qualityForced,
        )
    }

    /** contracts §10: translate free text only; guards re-run on the translation (digits+units); any failure → English only. */
    suspend fun translate(o: AnalysisOutput, lang: String): Localized? {
        val target = LANG_NAMES[lang] ?: return null
        val src = buildJsonObject {
            put("explanation", o.explanation)
            put("what_would_help", buildJsonArray { o.whatWouldHelp.forEach { add(JsonPrimitive(it)) } })
            put("self_care_info", buildJsonArray { o.selfCareInfo.forEach { add(JsonPrimitive(it)) } })
            put("category_why", buildJsonObject { o.possibleCategories.forEach { put(it.key, it.why) } })
        }
        val text = llm.generate(LlmTask.TRANSLATE, prompts.translateSystem(target), PromptBuilder.cj(src), null)
        val obj = OutputParser.extractJson(text)?.let { runCatching { SnJson.parseToJsonElement(it).jsonObject }.getOrNull() } ?: return null
        fun list(k: String) = (obj[k] as? JsonArray)?.mapNotNull { (it as? JsonPrimitive)?.content }
        val exp = (obj["explanation"] as? JsonPrimitive)?.content ?: return null
        val wh = list("what_would_help") ?: return null
        val sc = list("self_care_info") ?: return null
        val why = (obj["category_why"] as? JsonObject)?.mapValues { it.value.jsonPrimitive.content } ?: emptyMap()
        val all = listOf(exp) + wh + sc + why.values
        if (all.any { guards.check(it).isNotEmpty() }) return null
        if (wh.size != o.whatWouldHelp.size || sc.size != o.selfCareInfo.size) return null
        return Localized(lang, exp, wh, sc, why)
    }

    companion object {
        val LANG_NAMES = mapOf("hi" to "Hindi", "kn" to "Kannada", "ta" to "Tamil", "te" to "Telugu", "bn" to "Bengali", "mr" to "Marathi")
    }
}

/** Basic mode (architecture §3 Fallback renderer): CV top-3 + condition cards + rules. Deterministic, English. */
object Fallback {
    fun likelihood(rank: Int, p: Double) = when {
        rank == 0 && p >= 0.5 -> "higher"
        p >= 0.15 -> "possible"
        else -> "less_likely"
    }

    fun uncertainty(top1: Double?, forced: Boolean): String = when {
        forced || top1 == null -> "high"
        top1 >= 0.7 -> "low"
        top1 >= 0.4 -> "moderate"
        else -> "high"
    }

    private fun firstSentence(s: String): String = (Regex("""^[^.!?]*[.!?]""").find(s)?.value ?: s).take(200)

    fun build(a: QuestionnaireAnswers, top3: List<CvScore>, cards: JsonObject, forcedHigh: Boolean, tier: Tier): AnalysisOutput {
        val cats = top3.filter { it.p >= 0.05 }.ifEmpty { top3.take(1) }.mapIndexed { i, s ->
            val card = cards[s.key] as? JsonObject
            val summary = (card?.get("summary") as? JsonPrimitive)?.content ?: ""
            Category(s.key, likelihood(i, s.p), firstSentence(summary).ifEmpty { "Closest match from the image model." })
        }
        val top1 = top3.firstOrNull()
        val unc = uncertainty(top1?.p, forcedHigh)
        val reasons = buildList {
            add("Basic mode: written explanation not available")
            if (top1 != null && top1.p < 0.5) add("The image model is not confident")
            if (forcedHigh) add("Photo quality was poor")
        }
        val dur = mapOf("lt_1w" to "less than a week", "1_4w" to "1–4 weeks", "1_6m" to "1–6 months", "gt_6m" to "more than 6 months")[a.duration] ?: a.duration
        val explanation = "You said this has been there for $dur, with itch ${a.itch} of 3 and pain ${a.pain} of 3. " +
            "The categories above are the image model's closest matches, shown with general notes for each. " +
            "Read the 'see a doctor if' points for the top match and follow the advice level shown."
        val care = top1?.let { (cards[it.key] as? JsonObject)?.get("general_care")?.jsonArray?.map { e -> e.jsonPrimitive.content } } ?: emptyList()
        return AnalysisOutput(
            possibleCategories = cats,
            uncertainty = Uncertainty(unc, reasons),
            explanation = explanation,
            whatWouldHelp = listOf("A clear photo in daylight, 15–20 cm away", "A doctor's examination if you are unsure"),
            selfCareInfo = care,
            triage = Triage(tier.name, ""),
            disagreementWithImageModel = false,
        )
    }
}
