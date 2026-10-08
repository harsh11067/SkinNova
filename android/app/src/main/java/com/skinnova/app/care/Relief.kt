package com.skinnova.app.care

import com.skinnova.app.data.Profile
import com.skinnova.app.model.QuestionnaireAnswers
import com.skinnova.app.model.SnJson
import com.skinnova.app.model.Tier
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

/** assets/care/relief.json — curated, cited home care / food / pharmacy notes (never LLM text). */
@Serializable
data class Txt(val en: String, val hi: String) { fun get(lang: String) = if (lang == "hi") hi else en }

@Serializable
data class ReliefItem(
    val id: String, val en: String, val hi: String, val kind: String = "",
    /** condition-specific (antifungal, scabicide, steroid…): only when the top category is a clear lead */
    val specific: Boolean = false,
    /** not for children under 12 */
    val adult: Boolean = false,
    /** "ok" | "ask" (pregnant/breastfeeding → ask a pharmacist) | "avoid" (hidden when pregnant) */
    val pregnancy: String = "ok",
    /** hidden when any of these categories is among the listed possibilities (steroids worsen fungal rashes) */
    @SerialName("avoid_if") val avoidIf: List<String> = emptyList(),
    /** hidden when the profile lists one of these (Profile.CONDITIONS keys) */
    @SerialName("avoid_conditions") val avoidConditions: List<String> = emptyList(),
) { fun get(lang: String) = if (lang == "hi") hi else en }

@Serializable
data class CareCard(val home: List<Txt> = emptyList(), val food: List<Txt> = emptyList(), val pharmacy: List<ReliefItem> = emptyList(),
                    val sources: List<String> = emptyList(), val contagious: Txt? = null)

@Serializable
data class Symptom(val home: List<Txt> = emptyList(), val pharmacy: List<ReliefItem> = emptyList())

@Serializable
data class General(val itch: Symptom, val pain: Symptom, val sources: List<String> = emptyList())

@Serializable
data class ReliefDb(val version: String, val general: General, val cards: Map<String, CareCard>) {
    companion object { fun parse(json: String) = SnJson.decodeFromString(serializer(), json) }
}

/** Why the pharmacy part is not shown. */
enum class Blocked { DOCTOR_FIRST, CHILD }

/** Extra lines shown under the pharmacy list. */
enum class Caution { PHARMACIST, PREGNANT, ALLERGIES, INFECTION_RISK, CONFIRM_FIRST }

data class ReliefPlan(val home: List<Txt>, val food: List<Txt>, val pharmacy: List<ReliefItem>, val blocked: Blocked?,
                      val cautions: List<Caution>, val sources: List<String>, val contagious: Txt? = null)

/**
 * Deterministic (JVM-tested): which home care, food notes and pharmacy items a result may show.
 * - HIGH/URGENT advice or a suspicious spot → no self-treatment, "see a doctor first".
 * - A child (under 12) → no medicines listed; a pharmacist or doctor chooses child doses.
 * - Condition-specific items need a clear lead (likelihood "higher" and image score ≥ [CONFIDENT_P]); otherwise only
 *   symptom relief (itch / pain) + "a pharmacist can look at it".
 * - An item is dropped when a blocking category is listed (steroid + possible fungus), when pregnant and the item is
 *   "avoid", or when the profile lists a condition it must not be used with (ibuprofen + asthma/ulcer/kidney).
 */
object ReliefPlanner {
    const val CONFIDENT_P = 0.5

    fun plan(db: ReliefDb, topKey: String, topLikelihood: String, topP: Double, listed: Set<String>, tier: Tier,
             a: QuestionnaireAnswers, profile: Profile): ReliefPlan {
        val card = db.cards[topKey] ?: db.cards["other"] ?: CareCard()
        val doctorFirst = tier >= Tier.HIGH || topKey == "suspicious_lesion"
        val home = (card.home + if (!doctorFirst && a.itch >= 1) db.general.itch.home else emptyList()).distinct()
        val sources = (card.sources + if (a.itch >= 1 || a.pain >= 1) db.general.sources else emptyList()).distinct()
        val child = a.ageBand == "lt_12"
        if (doctorFirst || child)
            return ReliefPlan(home, card.food, emptyList(), if (doctorFirst) Blocked.DOCTOR_FIRST else Blocked.CHILD, emptyList(), sources, card.contagious)

        val confident = topLikelihood == "higher" && topP >= CONFIDENT_P
        val candidates = card.pharmacy.filter { !it.specific || confident } +
            (if (a.itch >= 1) db.general.itch.pharmacy else emptyList()) + (if (a.pain >= 1) db.general.pain.pharmacy else emptyList())
        val items = candidates.distinctBy { it.id }.filter { i ->
            i.avoidIf.none { it in listed } &&
                !(profile.pregnant && i.pregnancy == "avoid") &&
                i.avoidConditions.none { it in profile.conditions }
        }
        val cautions = buildList {
            add(Caution.PHARMACIST)
            if (!confident && card.pharmacy.any { it.specific }) add(Caution.CONFIRM_FIRST)
            if (profile.pregnant && items.any { it.pregnancy != "ok" }) add(Caution.PREGNANT)
            if (profile.allergies.isNotBlank() && items.isNotEmpty()) add(Caution.ALLERGIES)
            if ((("diabetes" in profile.conditions) || ("weak_immunity" in profile.conditions)) && topKey in INFECTIONS) add(Caution.INFECTION_RISK)
        }
        return ReliefPlan(home, card.food, items, null, cautions, sources, card.contagious)
    }

    private val INFECTIONS = setOf("tinea", "scabies", "other")
}
