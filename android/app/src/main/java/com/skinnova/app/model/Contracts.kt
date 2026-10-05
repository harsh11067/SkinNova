package com.skinnova.app.model

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json

/** Exact mirrors of docs/contracts.md (Python twin: ml/common/schema.py). */

val SnJson = Json { ignoreUnknownKeys = true; encodeDefaults = true; explicitNulls = false }

enum class Tier { LOW, MODERATE, HIGH, URGENT;
    companion object {
        fun parse(s: String?): Tier? = entries.firstOrNull { it.name == s }
        fun max(vararg t: Tier?): Tier = t.filterNotNull().maxByOrNull { it.ordinal } ?: LOW
    }
}

object Enums {
    val BODY_SITES = listOf("face", "scalp", "neck", "chest", "back", "abdomen", "arm", "hand", "leg", "foot", "groin", "nails", "other")
    val DURATIONS = listOf("lt_1w", "1_4w", "1_6m", "gt_6m")
    val CHANGING = listOf("no", "growing", "changing_color", "changing_shape", "spreading", "unsure")
    val AGE_BANDS = listOf("lt_12", "12_17", "18_39", "40_59", "60_plus")
    val SKIN_TONES = listOf("fitz_1_2", "fitz_3_4", "fitz_5_6", "unknown")
    val LIKELIHOODS = listOf("higher", "possible", "less_likely")
    val UNCERTAINTY = listOf("low", "moderate", "high")
}

@Serializable
data class QuestionnaireAnswers(
    @SerialName("body_site") val bodySite: String,
    val duration: String,
    val itch: Int,
    val pain: Int,
    val changing: String,
    @SerialName("bleeding_or_crusting") val bleedingOrCrusting: Boolean,
    @SerialName("fever_or_unwell") val feverOrUnwell: Boolean,
    @SerialName("others_affected") val othersAffected: Boolean,
    @SerialName("new_product_or_exposure") val newProductOrExposure: Boolean,
    @SerialName("age_band") val ageBand: String,
    @SerialName("skin_tone") val skinTone: String = "unknown",
    @SerialName("free_text") val freeText: String = "",
    val source: String = "tap",
)

@Serializable
data class CvScore(val key: String, val p: Double)

@Serializable
data class Category(val key: String, val likelihood: String, val why: String)

@Serializable
data class Uncertainty(val level: String, val reasons: List<String> = emptyList())

@Serializable
data class Triage(val tier: String, val advice: String)

@Serializable
data class AnalysisOutput(
    @SerialName("possible_categories") val possibleCategories: List<Category>,
    val uncertainty: Uncertainty,
    val explanation: String,
    @SerialName("what_would_help") val whatWouldHelp: List<String> = emptyList(),
    @SerialName("self_care_info") val selfCareInfo: List<String> = emptyList(),
    val triage: Triage,
    @SerialName("disagreement_with_image_model") val disagreementWithImageModel: Boolean = false,
)

@Serializable
data class Localized(
    val lang: String,
    val explanation: String,
    @SerialName("what_would_help") val whatWouldHelp: List<String> = emptyList(),
    @SerialName("self_care_info") val selfCareInfo: List<String> = emptyList(),
    @SerialName("category_why") val categoryWhy: Map<String, String> = emptyMap(),
)

enum class Mode { full, basic }

/** What the Result UI renders (contracts §3). */
@Serializable
data class FinalResult(
    val output: AnalysisOutput,
    @SerialName("cv_top3") val cvTop3: List<CvScore>,
    @SerialName("rule_messages") val ruleMessages: List<String>,
    @SerialName("rules_fired") val rulesFired: List<String>,
    @SerialName("final_tier") val finalTier: String,
    val mode: Mode,
    @SerialName("fallback_reason") val fallbackReason: String? = null,
    @SerialName("prompt_version") val promptVersion: String,
    @SerialName("model_sha") val modelSha: String,
    val lang: String = "en",
    val localized: Localized? = null,
    val answers: QuestionnaireAnswers,
    @SerialName("quality_forced") val qualityForced: Boolean = false,
    @SerialName("created_at") val createdAt: Long = System.currentTimeMillis(),
)

@Serializable
data class LabelClass(
    val id: Int, val key: String, val display: String,
    @SerialName("tier_floor") val tierFloor: String,
    @SerialName("lesion_type") val lesionType: Boolean,
)

@Serializable
data class Labels(val version: String, val classes: List<LabelClass>) {
    val keys get() = classes.map { it.key }
    fun byKey(k: String) = classes.firstOrNull { it.key == k }
}

/** Timeline inputs to T1–T3 (contracts §8 subset). */
@Serializable
data class TimelineRuleInput(
    @SerialName("align_ok") val alignOk: Boolean,
    @SerialName("coin_in_both") val coinInBoth: Boolean = false,
    @SerialName("area_ratio") val areaRatio: Double? = null,
    @SerialName("contrast_delta") val contrastDelta: Double? = null,
    @SerialName("noise_contrast") val noiseContrast: Double = 0.0,
    val confidence: String = "low",
    @SerialName("lesion_type") val lesionType: Boolean = false,
)

/** contracts §9 */
@Serializable
data class ExtractedField(val value: kotlinx.serialization.json.JsonElement, val evidence: String)
