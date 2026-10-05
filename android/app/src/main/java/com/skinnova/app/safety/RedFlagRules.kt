package com.skinnova.app.safety

import com.skinnova.app.model.CvScore
import com.skinnova.app.model.Labels
import com.skinnova.app.model.QuestionnaireAnswers
import com.skinnova.app.model.Tier
import com.skinnova.app.model.TimelineRuleInput

/**
 * Deterministic rules R1–R8, T1–T3 (contracts §4). Pure functions; Python twin: ml/eval/redflags.py.
 * Both run tests/fixtures/redflag_cases.json. Messages are strings.xml keys (rf_r1 …), never LLM text.
 */
data class RuleResult(
    val tier: Tier = Tier.LOW,
    val fired: List<String> = emptyList(),
    val messages: List<String> = emptyList(),
    val forceUncertaintyHigh: Boolean = false,
) {
    fun raiseTo(t: Tier, rule: String, msgKey: String?): RuleResult = copy(
        tier = Tier.max(tier, t),
        fired = if (rule in fired) fired else fired + rule,
        messages = if (msgKey == null || msgKey in messages) messages else messages + msgKey,
    )
}

object RedFlagRules {
    private val LESION_CHANGES = setOf("growing", "changing_color", "changing_shape")
    private val LONG_DURATIONS = setOf("1_4w", "1_6m", "gt_6m")
    const val CLASS_FLOOR_P = 0.15
    const val SUSPICIOUS_P = 0.15

    fun top3(cv: List<CvScore>) = cv.sortedByDescending { it.p }.take(3)

    fun evaluate(
        a: QuestionnaireAnswers, cv: List<CvScore>, labels: Labels,
        qualityForced: Boolean = false, timeline: TimelineRuleInput? = null,
    ): RuleResult {
        var r = RuleResult()
        val t3 = top3(cv)
        if (a.feverOrUnwell && (a.pain >= 2 || a.changing == "spreading")) r = r.raiseTo(Tier.URGENT, "R1", "rf_r1")
        if (a.changing in LESION_CHANGES && t3.any { labels.byKey(it.key)?.lesionType == true }) r = r.raiseTo(Tier.HIGH, "R2", "rf_r2")
        if (a.bleedingOrCrusting && a.duration in LONG_DURATIONS) r = r.raiseTo(Tier.HIGH, "R3", "rf_r3")
        val sus = cv.firstOrNull { it.key == "suspicious_lesion" }?.p ?: 0.0
        if (sus >= SUSPICIOUS_P) r = r.raiseTo(Tier.HIGH, "R4", "rf_r4")
        if (a.ageBand == "lt_12") r = r.raiseTo(Tier.MODERATE, "R5", "rf_r5")
        if (a.bodySite in setOf("groin", "face") && a.pain >= 2) r = r.raiseTo(Tier.MODERATE, "R6", "rf_r6")
        if (a.othersAffected && a.itch >= 2) r = r.raiseTo(Tier.MODERATE, "R7", "rf_r7")
        if (qualityForced && "R8" !in r.fired) r = r.copy(forceUncertaintyHigh = true, fired = r.fired + "R8", messages = r.messages + "rf_r8")
        if (timeline != null) r = applyTimeline(r, timeline)
        return r
    }

    /** T1–T3. confidence=low disables T1/T2 (contracts §8); T3 still allowed. */
    fun applyTimeline(r0: RuleResult, t: TimelineRuleInput): RuleResult {
        if (!t.alignOk) return r0
        var r = r0
        val confOk = t.confidence == "ok"
        val area = t.areaRatio
        val contrast = t.contrastDelta
        if (confOk && t.lesionType && t.coinInBoth && area != null && area >= 1.25) r = r.raiseTo(Tier.HIGH, "T1", "rf_t1")
        if (confOk && t.lesionType && contrast != null && contrast >= maxOf(5.0, 2 * t.noiseContrast)) r = r.raiseTo(Tier.HIGH, "T2", "rf_t2")
        if (!t.lesionType && area != null && area >= 1.5) r = r.raiseTo(Tier.MODERATE, "T3", "rf_t3")
        return r
    }
}

object TierResolver {
    /** tier_floor of any class in top-3 with calibrated p ≥ 0.15. */
    fun classFloor(cv: List<CvScore>, labels: Labels): Tier =
        RedFlagRules.top3(cv).filter { it.p >= RedFlagRules.CLASS_FLOOR_P }
            .mapNotNull { labels.byKey(it.key)?.let { c -> Tier.parse(c.tierFloor) } }
            .fold(Tier.LOW) { acc, t -> Tier.max(acc, t) }

    /** Monotone max: the LLM can raise but never lower. */
    fun resolve(llm: Tier?, rule: Tier, floor: Tier, timeline: Tier? = null): Tier = Tier.max(rule, floor, llm, timeline)
}
