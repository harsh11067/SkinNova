package com.skinnova.app.ui

import android.app.Application
import android.graphics.Bitmap
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.skinnova.app.SkinNovaApp
import com.skinnova.app.data.CaptureEntity
import com.skinnova.app.data.MetricsEntity
import com.skinnova.app.data.SpotEntity
import com.skinnova.app.ml.LlmTask
import com.skinnova.app.ml.PromptBuilder
import com.skinnova.app.ml.toRgb
import com.skinnova.app.ml.validateNarration
import com.skinnova.app.model.CvScore
import com.skinnova.app.model.SnJson
import com.skinnova.app.model.TimelineRuleInput
import com.skinnova.app.reminders.ReminderWorker
import com.skinnova.app.safety.RedFlagRules
import com.skinnova.app.safety.RuleResult
import com.skinnova.app.timeline.ChangeMetrics
import com.skinnova.app.timeline.NoiseFloor
import com.skinnova.app.timeline.Timeline
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlinx.serialization.builtins.ListSerializer
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import java.text.DateFormat
import java.util.Date
import java.util.UUID
import java.util.concurrent.TimeUnit

sealed interface RecaptureState {
    data object Idle : RecaptureState
    data object Measuring : RecaptureState
    data object Narrating : RecaptureState
    data class Done(val metrics: ChangeMetrics) : RecaptureState
    data object NoMatch : RecaptureState
    data class Error(val msg: String) : RecaptureState
}

class TimelineViewModel(app: Application) : AndroidViewModel(app) {
    private val c = (app as SkinNovaApp).container
    private val _rec = MutableStateFlow<RecaptureState>(RecaptureState.Idle); val rec: StateFlow<RecaptureState> = _rec
    private val _baseline = MutableStateFlow<Bitmap?>(null); val baseline: StateFlow<Bitmap?> = _baseline
    /** The spot's tap on the baseline (normalised): the live coin badge must not take the spot itself for the coin. */
    private val _seed = MutableStateFlow<Pair<Double, Double>?>(null); val seed: StateFlow<Pair<Double, Double>?> = _seed

    private fun cvJson(cv: List<CvScore>) = SnJson.encodeToString(ListSerializer(CvScore.serializer()), cv)
    private fun cvFrom(s: String) = SnJson.decodeFromString(ListSerializer(CvScore.serializer()), s)

    /** "Track this spot": baseline = the analysed photo; seed = user's tap. Timeline requires opt-in (architecture §9). */
    fun createSpot(name: String, bodySite: String, seedX: Float, seedY: Float, reminderDays: Int, photo: Bitmap, cv: List<CvScore>,
                   analysisId: String?, onCreated: (String) -> Unit) = viewModelScope.launch {
        val spotId = UUID.randomUUID().toString(); val capId = UUID.randomUUID().toString()
        val ref = c.images.save(photo)
        val lesion = cv.take(3).any { c.labels.byKey(it.key)?.lesionType == true && it.p >= 0.15 }
        c.db.dao().put(SpotEntity(spotId, name, bodySite, seedX, seedY, c.settings.coinMm.value, reminderDays, System.currentTimeMillis(), capId, lesion))
        c.db.dao().put(CaptureEntity(capId, spotId, ref, System.currentTimeMillis(), cvJson(cv), analysisId))
        ReminderWorker.schedule(getApplication(), spotId, name, reminderDays)
        onCreated(spotId)
    }

    fun loadBaseline(spotId: String) = viewModelScope.launch {
        val s = c.db.dao().spot(spotId) ?: return@launch
        _seed.value = s.seedX.toDouble() to s.seedY.toDouble()
        val cap = s.baselineCaptureId?.let { c.db.dao().capture(it) } ?: return@launch
        _baseline.value = c.images.load(cap.imageRef)
    }

    fun reset() { _rec.value = RecaptureState.Idle }

    /** Re-capture → align → metrics → rules T1–T3 → narration (validated; deterministic fallback text otherwise). */
    fun recapture(spotId: String, photo: Bitmap, calibration: Boolean) = viewModelScope.launch {
        _rec.value = RecaptureState.Measuring
        try {
            val s = c.db.dao().spot(spotId)!!
            val baseCap = c.db.dao().capture(s.baselineCaptureId!!)!!
            val base = c.images.load(baseCap.imageRef)!!
            val cvNew = withContext(Dispatchers.Default) { if (c.cv.available) c.cv.classify(photo.toRgb()) else emptyList() }
            val cvBase = cvFrom(baseCap.cvProbsJson)
            val keys = c.labels.keys
            fun vec(l: List<CvScore>) = DoubleArray(keys.size) { i -> l.firstOrNull { it.key == keys[i] }?.p ?: 0.0 }
            val noise = s.noiseN.takeIf { it >= 3 }?.let { NoiseFloor(s.noiseArea ?: 0.0, s.noiseContrast ?: 0.0, it) }
            var m = withContext(Dispatchers.Default) {
                Timeline.compute(base, photo, s.seedX.toDouble(), s.seedY.toDouble(), s.coinDiameterMm, if (cvBase.isEmpty()) null else vec(cvBase),
                    if (cvNew.isEmpty()) null else vec(cvNew), noise, s.lesionType)
            }
            val capId = UUID.randomUUID().toString()
            val ref = c.images.save(photo)
            c.db.dao().put(CaptureEntity(capId, spotId, ref, System.currentTimeMillis(), cvJson(cvNew), null, calibration))
            if (!m.alignOk) { _rec.value = RecaptureState.NoMatch; return@launch }
            val days = TimeUnit.MILLISECONDS.toDays(System.currentTimeMillis() - baseCap.takenAt).toInt()
            val r: RuleResult = RedFlagRules.applyTimeline(RuleResult(), TimelineRuleInput(m.alignOk, m.coinInBoth, m.areaRatio, m.contrastDelta,
                noise?.contrast ?: 0.0, m.confidence, s.lesionType))
            m = m.copy(captureId = capId, baselineCaptureId = baseCap.id, daysSinceBaseline = days, timelineTier = r.tier.name, triggeredRules = r.fired)
            if (calibration) updateNoiseFloor(s)
            _rec.value = RecaptureState.Narrating
            val baseDate = DateFormat.getDateInstance(DateFormat.MEDIUM).format(Date(baseCap.takenAt))
            val narration = narrate(m, cvBase, cvNew, baseDate)
            c.db.dao().put(MetricsEntity(capId, baseCap.id, m.json(), r.tier.name, narration))
            _rec.value = RecaptureState.Done(m)
        } catch (e: Throwable) { _rec.value = RecaptureState.Error(e.message ?: e.javaClass.simpleName) }
    }

    private suspend fun updateNoiseFloor(s: SpotEntity) {
        val caps = c.db.dao().capturesNow(s.id).filter { it.calibration || it.id == s.baselineCaptureId }
        val bmps = caps.mapNotNull { c.images.load(it.imageRef) }
        val nf = withContext(Dispatchers.Default) { Timeline.noiseFloor(bmps, s.seedX.toDouble(), s.seedY.toDouble()) } ?: return
        c.db.dao().put(s.copy(noiseArea = nf.area, noiseContrast = nf.contrast, noiseN = nf.n))
    }

    /** contracts §8 narration: LLM (2–4 sentences, guards, must state low confidence) else deterministic text. */
    private suspend fun narrate(m: ChangeMetrics, before: List<CvScore>, after: List<CvScore>, baseDate: String): String {
        if (c.llm.available) runCatching {
            val metrics = buildJsonObject {
                put("days_since_baseline", m.daysSinceBaseline); put("align_score", m.alignScore); put("align_ok", m.alignOk)
                put("coin_in_both", m.coinInBoth); m.areaRatio?.let { put("area_ratio", Math.round(it * 100) / 100.0) }
                m.contrastDelta?.let { put("contrast_delta", Math.round(it * 10) / 10.0) }; put("confidence", m.confidence); put("lesion_type", m.lesionType)
            }
            fun t3(l: List<CvScore>) = buildJsonArray { PromptBuilder.top3(l).forEach { s -> add(buildJsonObject { put("key", s.key); put("p", s.p) }) } }
            val user = c.prompts.template("narrate_user.txt").second.replace("{baseline_date}", baseDate)
                .replace("{metrics_json}", PromptBuilder.cj(metrics)).replace("{cv_before_json}", PromptBuilder.cj(t3(before)))
                .replace("{cv_after_json}", PromptBuilder.cj(t3(after))).replace("{timeline_tier}", m.timelineTier)
            val out = c.llm.generate(LlmTask.NARRATE, c.prompts.narrateSystem(), user).trim()
            if (validateNarration(out, c.guards, m.confidence == "low").isEmpty()) return out
        }
        return fallbackNarration(m, baseDate)
    }

    companion object {
        fun fallbackNarration(m: ChangeMetrics, baseDate: String): String {
            val sb = StringBuilder()
            val pct = m.areaRatio?.let { Math.round(kotlin.math.abs(it - 1) * 100) }
            sb.append(when {
                pct == null -> "The size could not be measured."
                pct < 5 -> "Compared with $baseDate, the measured area is about the same."
                else -> "Compared with $baseDate, the measured area is about $pct percent ${if (m.areaRatio!! > 1) "larger" else "smaller"}."
            })
            m.contrastDelta?.let { sb.append(if (kotlin.math.abs(it) < 1.5) " Its colour contrast with the surrounding skin barely changed."
                else " Its colour contrast with the surrounding skin ${if (it > 0) "increased" else "decreased"} by ${"%.1f".format(kotlin.math.abs(it))} units.") }
            if (m.confidence == "low") sb.append(" This measurement is uncertain because the photos lack a coin for scale or calibration.")
            val big = (m.areaRatio ?: 1.0) >= 1.25 || kotlin.math.abs(m.contrastDelta ?: 0.0) >= 5
            sb.append(when {
                m.timelineTier == "HIGH" -> " Please show this spot to a doctor within a few days."
                m.timelineTier == "MODERATE" -> " It is worth showing this change to a doctor soon."
                m.confidence == "low" && big -> " Retake the photo with a coin to measure it properly, and show a doctor if it really seems to be growing or darkening."
                else -> " Nothing here suggests a worrying change, but keep tracking it."
            })
            return sb.toString()
        }
    }
}
