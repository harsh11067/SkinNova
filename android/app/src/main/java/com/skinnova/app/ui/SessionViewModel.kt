package com.skinnova.app.ui

import android.app.Application
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.net.Uri
import androidx.exifinterface.media.ExifInterface
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.skinnova.app.SkinNovaApp
import com.skinnova.app.data.AnalysisEntity
import com.skinnova.app.ml.AnalysisState
import com.skinnova.app.ml.EngineHolder
import com.skinnova.app.ml.IntakeResult
import com.skinnova.app.ml.IntakeValidator
import com.skinnova.app.ml.LlmTask
import com.skinnova.app.ml.ModelMissing
import com.skinnova.app.ml.Quality
import com.skinnova.app.ml.QualityGate
import com.skinnova.app.ml.toRgb
import com.skinnova.app.ml.toBitmap
import com.skinnova.app.ml.Rgb
import com.skinnova.app.ml.ImageOps
import com.skinnova.app.model.CvScore
import com.skinnova.app.model.FinalResult
import com.skinnova.app.model.QuestionnaireAnswers
import com.skinnova.app.model.SnJson
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.util.UUID

/** Draft answers: null = not answered yet (voice intake never fills a field it did not hear). */
data class Draft(
    val bodySite: String? = null, val duration: String? = null, val itch: Int? = null, val pain: Int? = null,
    val changing: String? = null, val bleeding: Boolean? = null, val fever: Boolean? = null, val others: Boolean? = null,
    val product: Boolean? = null, val ageBand: String? = null, val skinTone: String = "unknown", val freeText: String = "",
    val evidence: Map<String, String> = emptyMap(), val viaVoice: Boolean = false,
) {
    fun complete() = listOf(bodySite, duration, itch, pain, changing, bleeding, fever, others, product, ageBand).all { it != null }
    fun toAnswers() = QuestionnaireAnswers(bodySite!!, duration!!, itch!!, pain!!, changing!!, bleeding!!, fever!!, others!!, product!!,
        ageBand!!, skinTone, freeText.take(200), if (viaVoice) "voice_confirmed" else "tap")
}

sealed interface VoiceState {
    data object Idle : VoiceState
    data class Recording(val level: Float, val ms: Long) : VoiceState
    data object Transcribing : VoiceState
    data class Heard(val transcript: String) : VoiceState
    data object Extracting : VoiceState
    data class Filled(val n: Int, val dropped: Int) : VoiceState
    data class Error(val kind: String) : VoiceState
}

/** True while the app lock covers the screens: dialogs must not open above the lock screen. */
val LocalAppLocked = androidx.compose.runtime.staticCompositionLocalOf { false }

/** Early look while the model writes: image-model top 3 + the safety rules' advice level (final level can only go up). */
data class EarlyLook(val top3: List<com.skinnova.app.model.CvScore>, val tier: com.skinnova.app.model.Tier, val ruleMessages: List<String>)

/** v2.1 item 3: honest progress while Gemma works. [fraction] null = indeterminate; [etaSec] null = unknown yet. */
enum class GenPhase { LOADING, READING, WRITING, CHECKING, TRANSLATING }
data class GenProgress(val phase: GenPhase, val fraction: Float? = null, val etaSec: Int? = null)

/** Expected output length of an analysis (shipped model, llm_val: median 1,027 chars, p90 1,219). */
const val EXPECTED_CHARS = 1100

/** Remaining-time estimate from the writing speed of the last ~5 s (JVM-tested). */
class EtaEstimator(private val expected: Int = EXPECTED_CHARS, private val windowMs: Long = 5_000) {
    private val pts = ArrayDeque<Pair<Long, Int>>()
    fun add(tMs: Long, chars: Int) { pts.addLast(tMs to chars); while (pts.size > 2 && tMs - pts.first().first > windowMs) pts.removeFirst() }
    fun reset() = pts.clear()
    /** null until there is a rate to go by. */
    fun etaSec(): Int? {
        if (pts.size < 2) return null
        val (t0, c0) = pts.first(); val (t1, c1) = pts.last()
        val rate = (c1 - c0) / ((t1 - t0) / 1000.0); if (rate <= 0.0) return null
        return ((expected - c1).coerceAtLeast(0) / rate).toInt().coerceAtLeast(1)
    }
    fun fraction(chars: Int) = (chars.toFloat() / expected).coerceIn(0f, 0.97f)
}

/** One Ask SkinNova exchange. [answer] null while the model is writing. */
data class ChatTurn(val question: String, val answer: String? = null, val danger: Boolean = false, val withheld: Boolean = false,
                    val partial: String = "")

class SessionViewModel(app: Application) : AndroidViewModel(app) {
    val c = (app as SkinNovaApp).container

    private val _photo = MutableStateFlow<Bitmap?>(null); val photo: StateFlow<Bitmap?> = _photo
    private val _quality = MutableStateFlow<Quality?>(null); val quality: StateFlow<Quality?> = _quality
    var qualityForced = false
    private val _draft = MutableStateFlow(Draft()); val draft: StateFlow<Draft> = _draft
    private val _state = MutableStateFlow<AnalysisState>(AnalysisState.Idle); val state: StateFlow<AnalysisState> = _state
    private val _result = MutableStateFlow<FinalResult?>(null); val result: StateFlow<FinalResult?> = _result
    private val _cv = MutableStateFlow<List<CvScore>>(emptyList()); val cv: StateFlow<List<CvScore>> = _cv
    private val _voice = MutableStateFlow<VoiceState>(VoiceState.Idle); val voice: StateFlow<VoiceState> = _voice
    private val _chat = MutableStateFlow<List<ChatTurn>>(emptyList()); val chat: StateFlow<List<ChatTurn>> = _chat
    private val _early = MutableStateFlow<EarlyLook?>(null); val early: StateFlow<EarlyLook?> = _early
    /** true while the early result is on screen and Gemma is still writing the full one */
    private val _pending = MutableStateFlow(false); val pending: StateFlow<Boolean> = _pending
    private val _progress = MutableStateFlow<GenProgress?>(null); val progress: StateFlow<GenProgress?> = _progress
    /** the full result's leading category differs from the early one → "Updated after reading your answers" */
    private val _updatedTop = MutableStateFlow(false); val updatedTop: StateFlow<Boolean> = _updatedTop
    /** consecutive results whose leading category is "other" (item 8: a 2nd one recommends a doctor) */
    var otherStreak = 0; private set
    private val eta = EtaEstimator()
    var savedId: String? = null; private set
    private var job: Job? = null
    private val audio = com.skinnova.app.ml.AudioCapture()

    // ---------- photo ----------
    /** A new scan or an opened history item replaces a result that is still being written: finish the old one as a
     *  Basic result (and its history row) first, so it can never land on the new screen later. */
    private fun abandonRunning() {
        if (job?.isActive == true) { if (_pending.value) stopWriting(); job?.cancel(); _pending.value = false; _progress.value = null }
    }

    fun setPhoto(bmp: Bitmap) {
        abandonRunning()
        val b = downscale(bmp, 1024)
        // profile pre-fills age band + skin tone (shown on the questions, the user can still change them)
        val prof = c.profiles.profile.value
        _photo.value = b; qualityForced = false; _result.value = null; savedId = null; _cv.value = emptyList(); _chat.value = emptyList(); _early.value = null
        _draft.value = Draft(ageBand = prof.ageBand, skinTone = prof.skinTone)
        warmUp()   // v2.1 item 5: load the model while the user reviews the photo and answers (not only on the questions)
        viewModelScope.launch(Dispatchers.Default) {
            val rgb = b.toRgb(); val q = QualityGate.check(rgb)
            // image model now (not at analysis time): its skin-photo gate warns before the questions; scores are reused
            val cvOut = if (c.cv.available) runCatching { c.cv.classifyWithSkin(rgb) }.getOrNull() else null
            _cv.value = cvOut?.scores ?: emptyList()
            _quality.value = if (cvOut != null && !c.cv.looksLikeSkin(cvOut.pSkin)) q.copy(issues = q.issues + Quality.Issue.NOT_SKIN) else q
        }
    }

    fun setPhotoFromUri(uri: Uri) = viewModelScope.launch(Dispatchers.IO) {
        val bmp = decodeUpright(getApplication(), uri) ?: return@launch
        withContext(Dispatchers.Main) { setPhoto(bmp) }
    }

    fun forceQuality() { qualityForced = true }

    // ---------- questionnaire ----------
    fun update(f: (Draft) -> Draft) { _draft.value = f(_draft.value) }

    /** Start loading the LLM while the user answers (architecture §4 warm-up). */
    fun warmUp() {
        if (!c.models.let { it.activeModelPath() != null } || c.engineHolder.isLoaded) return
        viewModelScope.launch { runCatching { c.engineHolder.use { } } }
    }

    // ---------- analysis ----------
    fun analyze() {
        val bmp = _photo.value ?: return
        val answers = _draft.value.toAnswers()
        job?.cancel()
        val app = getApplication<Application>()
        com.skinnova.app.notify.Notifier.cancelReady(app)
        com.skinnova.app.notify.AnalysisService.start(app)   // foreground priority while the model works (user may switch apps)
        job = viewModelScope.launch {
            try {
                _state.value = AnalysisState.CheckingPhoto
                val rgb = withContext(Dispatchers.Default) { bmp.toRgb() }
                if (_quality.value == null) _quality.value = withContext(Dispatchers.Default) { QualityGate.check(rgb) }
                _state.value = AnalysisState.Classifying
                if (!c.cv.available) { _state.value = AnalysisState.Failed("Image model not found in this build"); return@launch }
                val cv = _cv.value.ifEmpty { withContext(Dispatchers.Default) { c.cv.classify(rgb) } }
                _cv.value = cv
                // early look (same deterministic rules the pipeline runs; the LLM can only raise the level)
                val rules = com.skinnova.app.safety.RedFlagRules.evaluate(answers, cv, c.labels, qualityForced)
                _early.value = EarlyLook(com.skinnova.app.safety.RedFlagRules.top3(cv),
                    com.skinnova.app.model.Tier.max(rules.tier, com.skinnova.app.safety.TierResolver.classFloor(cv, c.labels)), rules.messages)
                // the photo reaches the LLM only as the image model's scores (EngineHolder.SEND_PHOTO_TO_LLM)
                val img = if (EngineHolder.SEND_PHOTO_TO_LLM) withContext(Dispatchers.IO) { writeLlmImage(rgb) }.path else null
                var basic: FinalResult? = null
                val loadStart = System.currentTimeMillis(); val expectLoad = expectedLoadMs()
                val res0 = c.pipeline().run(answers, cv, qualityForced, img, c.settings.lang.value) { st ->
                    when (st) {
                        is AnalysisState.Preliminary -> {   // result first: the screen switches to it now
                            basic = st.result; _result.value = st.result; _pending.value = true; _updatedTop.value = false
                            if (c.settings.history.value) saveOrUpdate(st.result)   // a kill no longer loses the result
                        }
                        AnalysisState.LoadingModel -> _progress.value = GenProgress(GenPhase.LOADING,
                            ((System.currentTimeMillis() - loadStart).toFloat() / expectLoad).coerceIn(0f, 0.95f), null)
                        is AnalysisState.Generating -> {
                            val now = System.currentTimeMillis()
                            if (st.partialText.isEmpty()) { eta.reset(); _progress.value = GenProgress(GenPhase.READING) }
                            else { eta.add(now, st.partialText.length)
                                _progress.value = GenProgress(GenPhase.WRITING, eta.fraction(st.partialText.length), eta.etaSec()) }
                        }
                        AnalysisState.Validating -> _progress.value = GenProgress(GenPhase.CHECKING, 0.98f, null)
                        AnalysisState.Translating -> _progress.value = GenProgress(GenPhase.TRANSLATING)
                        else -> {}
                    }
                    _state.value = st
                }
                // the full result replaces the early one in place; its advice level can only be the same or higher
                val res = basic?.let { b -> res0.copy(finalTier = com.skinnova.app.model.Tier.max(com.skinnova.app.model.Tier.parse(res0.finalTier),
                    com.skinnova.app.model.Tier.parse(b.finalTier)).name) } ?: res0
                _updatedTop.value = basic != null && res.mode == com.skinnova.app.model.Mode.full &&
                    basic!!.output.possibleCategories.firstOrNull()?.key != res.output.possibleCategories.firstOrNull()?.key
                otherStreak = if (res.output.possibleCategories.firstOrNull()?.key == "other") otherStreak + 1 else 0
                _result.value = res; _pending.value = false; _progress.value = null
                if (c.settings.history.value) saveOrUpdate(res)
                _state.value = AnalysisState.Done(res)
                com.skinnova.app.notify.Notifier.ready(app, com.skinnova.app.model.Tier.parse(res.finalTier) ?: com.skinnova.app.model.Tier.MODERATE,
                    res.mode == com.skinnova.app.model.Mode.basic)
            } catch (e: kotlinx.coroutines.CancellationException) {
                stopWriting(); _state.value = AnalysisState.Idle; throw e
            } catch (e: Throwable) {
                // the early result stays (as a Basic result) if Gemma failed after it was shown
                if (_pending.value) stopWriting()
                _state.value = AnalysisState.Failed(e.message ?: e.javaClass.simpleName)
            } finally {
                _pending.value = false; _progress.value = null
                com.skinnova.app.notify.AnalysisService.stop(app)
            }
        }
    }

    /** Early result becomes the final (Basic) result: user pressed Stop, or the LLM failed. */
    private fun stopWriting() {
        val r = _result.value ?: return
        if (r.fallbackReason == com.skinnova.app.ml.AnalysisPipeline.PENDING) {
            val fin = r.copy(fallbackReason = com.skinnova.app.ml.AnalysisPipeline.STOPPED)
            _result.value = fin
            if (c.settings.history.value) saveOrUpdate(fin)
        }
    }

    fun cancel() { job?.cancel(); _state.value = AnalysisState.Idle }

    /** Engine load estimate for the progress bar: ~15 s with the GPU cache, ~2 min when it must be built. */
    private fun expectedLoadMs(): Long =
        if (EngineHolder.engineCacheDir(getApplication()).listFiles()?.any { it.name.contains("mldrift_weight_cache") } == true) 16_000L else 120_000L

    /** Forget the current photo, answers, result and chat (PIN reset / delete everything). */
    fun clearSession() {
        job?.cancel(); chatJob?.cancel()
        _photo.value = null; _result.value = null; _cv.value = emptyList(); _chat.value = emptyList(); _draft.value = Draft()
        _quality.value = null; _state.value = AnalysisState.Idle; savedId = null; _early.value = null
    }

    /** First call stores (photo + result); later calls update the same history row (early → full result). */
    private fun saveOrUpdate(res: FinalResult) {
        val id = savedId ?: return save(res)
        viewModelScope.launch {
            c.db.dao().analysis(id)?.let { e -> c.db.dao().put(e.copy(resultJson = SnJson.encodeToString(FinalResult.serializer(), res),
                topKey = res.output.possibleCategories.first().key, tier = res.finalTier, mode = res.mode.name)) }
        }
    }

    fun save(r: FinalResult? = _result.value) {
        val res = r ?: return; val bmp = _photo.value ?: return
        if (savedId != null) return
        val id = UUID.randomUUID().toString(); savedId = id
        viewModelScope.launch {
            val ref = c.images.save(bmp)
            c.db.dao().put(AnalysisEntity(id, res.createdAt, ref, SnJson.encodeToString(FinalResult.serializer(), res),
                res.output.possibleCategories.first().key, res.finalTier, res.mode.name))
        }
    }

    fun openSaved(e: AnalysisEntity) = viewModelScope.launch {
        abandonRunning()
        // an early result saved before the app was closed mid-writing shows as stopped, not as "still writing"
        _result.value = SnJson.decodeFromString(FinalResult.serializer(), e.resultJson).let {
            if (it.fallbackReason == com.skinnova.app.ml.AnalysisPipeline.PENDING) it.copy(fallbackReason = com.skinnova.app.ml.AnalysisPipeline.STOPPED) else it }
        _pending.value = false; _updatedTop.value = false
        _photo.value = c.images.load(e.imageRef); savedId = e.id; _chat.value = emptyList()
        _cv.value = _result.value!!.cvTop3
    }

    // ---------- Ask SkinNova: follow-up questions about the result (on-device Gemma) ----------
    private var chatJob: Job? = null
    val chatBusy get() = chatJob?.isActive == true

    fun ask(question: String) {
        val q = question.trim().take(300); val r = _result.value ?: return
        if (q.isEmpty() || chatBusy || !c.llm.available) return
        val danger = com.skinnova.app.safety.ChatSafety.dangerSigns(q)
        val earlier = _chat.value.mapNotNull { t -> t.answer?.takeIf { !t.withheld }?.let { t.question to it } }
        _chat.value = _chat.value + ChatTurn(q, danger = danger)
        fun setLast(f: (ChatTurn) -> ChatTurn) { _chat.value = _chat.value.dropLast(1) + f(_chat.value.last()) }
        chatJob = viewModelScope.launch {
            val ans = try {
                val top = r.output.possibleCategories.firstOrNull()?.key ?: "other"
                val card = c.relief.cards[top] ?: c.relief.cards["other"]
                // answer in the question's language (a Hindi question gets a Hindi answer, whatever the app language)
                val lang = if (q.any { it in '\u0900'..'\u097F' }) "hi" else c.settings.lang.value
                val p = c.prompts.chat(r, earlier, q, lang, card?.home?.map { it.en } ?: emptyList(), card?.food?.map { it.en } ?: emptyList(),
                    card?.contagious?.en)
                c.llm.generate(LlmTask.CHAT, p.system, p.user) { part -> setLast { it.copy(partial = part) } }.trim()
            } catch (e: kotlinx.coroutines.CancellationException) { throw e } catch (e: Throwable) { "" }
            val shown = com.skinnova.app.safety.ChatSafety.tidy(ans, q)
            val why = com.skinnova.app.safety.ChatSafety.withheld(shown, c.guards)
            setLast { it.copy(answer = if (why == null) shown else "", withheld = why != null, partial = "") }
        }
    }

    // ---------- on-demand Hindi (fine-tuned TRANSLATE task) ----------
    private val _translating = MutableStateFlow(false); val translating: StateFlow<Boolean> = _translating

    /** Translate this result into Hindi on the phone (contracts §10: guards re-run; any failure → stays English). */
    fun translateResult(onFail: () -> Unit) {
        val r = _result.value ?: return
        if (_translating.value || r.localized != null || !c.llm.available) return
        _translating.value = true
        viewModelScope.launch {
            val loc = runCatching { c.pipeline().translate(r.output, "hi") }.getOrNull()
            _translating.value = false
            if (loc == null) { onFail(); return@launch }
            val upd = r.copy(localized = loc, lang = "hi"); _result.value = upd
            savedId?.let { id ->   // keep the saved copy in step
                c.db.dao().analysis(id)?.let { e -> c.db.dao().put(e.copy(resultJson = SnJson.encodeToString(FinalResult.serializer(), upd))) }
            }
        }
    }

    /** Ask SkinNova by voice: the phone's on-device speech recogniser (the model file has no audio encoder). */
    suspend fun listenQuestion(onLevel: (Float) -> Unit): String? = try {
        com.skinnova.app.ml.OnDeviceSpeech.listen(getApplication(), c.settings.lang.value, onLevel).trim().takeIf { it.count { ch -> ch.isLetter() } >= 3 }
    } catch (e: kotlinx.coroutines.CancellationException) { throw e } catch (e: Throwable) { null }

    // ---------- voice intake (USP-2) ----------
    fun startRecording() {
        if (c.models.activeModelPath() == null) { _voice.value = VoiceState.Error("model"); return }
        viewModelScope.launch {
            // Custom exports may lose Gemma's audio encoder (plan §6 risk): then transcribe with the on-device recogniser.
            runCatching { c.engineHolder.use { } }
            if (!c.engineHolder.supportsAudio) {
                if (!com.skinnova.app.ml.OnDeviceSpeech.available(getApplication())) { _voice.value = VoiceState.Error("speech_pack"); return@launch }
                _voice.value = VoiceState.Recording(0f, 0)
                val t = try {
                    com.skinnova.app.ml.OnDeviceSpeech.listen(getApplication(), c.settings.lang.value) { l -> _voice.value = VoiceState.Recording(l, 0) }
                } catch (e: com.skinnova.app.ml.SpeechPackMissing) { _voice.value = VoiceState.Error("speech_pack"); return@launch }
                _voice.value = if (t.count { it.isLetter() } < 3) VoiceState.Error("unclear") else VoiceState.Heard(t)
                return@launch
            }
            try {
                val wav = audio.record({ l -> _voice.value = VoiceState.Recording(l, (_voice.value as? VoiceState.Recording)?.ms ?: 0) },
                    { ms -> _voice.value = VoiceState.Recording((_voice.value as? VoiceState.Recording)?.level ?: 0f, ms) })
                if (wav.size <= 44 + 16000) { _voice.value = VoiceState.Error("unclear"); return@launch }
                _voice.value = VoiceState.Transcribing
                val lang = c.settings.lang.value
                val t = c.llm.generate(LlmTask.TRANSCRIBE, "You transcribe speech exactly.", c.prompts.transcribe(if (lang == "hi") "Hindi or Hinglish" else "English or Hindi"),
                    audio = wav).trim()
                _voice.value = if (t.length < 3 || t.count { it.isLetter() } < 3) VoiceState.Error("unclear") else VoiceState.Heard(t)
            } catch (e: ModelMissing) { _voice.value = VoiceState.Error("model") } catch (e: Throwable) { _voice.value = VoiceState.Error("unclear") }
        }
    }

    fun stopRecording() = audio.stop()

    fun extract(transcript: String) = viewModelScope.launch {
        _voice.value = VoiceState.Extracting
        try {
            val user = c.prompts.template("extract_user.txt").second.replace("{transcript}", transcript.replace("<<<", "").replace(">>>", ""))
            val out = c.llm.generate(LlmTask.EXTRACT, c.prompts.extractSystem(), user)
            val r = IntakeValidator.validate(out, transcript, c.intakeTopics)
            val n = applyIntake(r, transcript)
            _voice.value = VoiceState.Filled(n, r.dropped.size)
        } catch (e: Throwable) { _voice.value = VoiceState.Error("unclear") }
    }

    /** Pre-fill ONLY validated fields; the user still confirms each one on the questionnaire. */
    fun applyIntake(r: IntakeResult, transcript: String): Int {
        if (!r.ok) return 0
        val f = r.fields
        var d = _draft.value
        val ev = d.evidence.toMutableMap()
        fun has(k: String) = f[k] != null
        IntakeValidator.strValue(f["body_site"])?.let { d = d.copy(bodySite = it); ev["body_site"] = f["body_site"]!!.evidence }
        IntakeValidator.strValue(f["duration"])?.let { d = d.copy(duration = it); ev["duration"] = f["duration"]!!.evidence }
        IntakeValidator.intValue(f["itch"])?.let { d = d.copy(itch = it); ev["itch"] = f["itch"]!!.evidence }
        IntakeValidator.intValue(f["pain"])?.let { d = d.copy(pain = it); ev["pain"] = f["pain"]!!.evidence }
        IntakeValidator.strValue(f["changing"])?.let { d = d.copy(changing = it); ev["changing"] = f["changing"]!!.evidence }
        IntakeValidator.boolValue(f["bleeding_or_crusting"])?.let { d = d.copy(bleeding = it); ev["bleeding_or_crusting"] = f["bleeding_or_crusting"]!!.evidence }
        IntakeValidator.boolValue(f["fever_or_unwell"])?.let { d = d.copy(fever = it); ev["fever_or_unwell"] = f["fever_or_unwell"]!!.evidence }
        IntakeValidator.boolValue(f["others_affected"])?.let { d = d.copy(others = it); ev["others_affected"] = f["others_affected"]!!.evidence }
        IntakeValidator.boolValue(f["new_product_or_exposure"])?.let { d = d.copy(product = it); ev["new_product_or_exposure"] = f["new_product_or_exposure"]!!.evidence }
        IntakeValidator.strValue(f["age_band"])?.let { d = d.copy(ageBand = it); ev["age_band"] = f["age_band"]!!.evidence }
        val notes = listOf(d.freeText, r.unparsedNotes).filter { it.isNotBlank() }.joinToString("; ").take(200)
        _draft.value = d.copy(evidence = ev, viaVoice = true, freeText = notes)
        return IntakeValidator.FIELD_NAMES.count { has(it) }
    }

    fun resetVoice() { _voice.value = VoiceState.Idle }

    /** Image handed to Gemma = the training images' preprocessing (ml/data/normalize.py): long side 512 with Pillow's
     *  LANCZOS (Kotlin port in ImageOps), JPEG quality 95. Gemma's own vision preprocessing then fits it to ≤ 2,520 patches. */
    private fun writeLlmImage(rgb: Rgb): File {
        val f = File(getApplication<Application>().cacheDir, "llm_input.jpg")
        val b = ImageOps.normalizeLongSide(rgb).toBitmap()
        f.outputStream().use { b.compress(Bitmap.CompressFormat.JPEG, 95, it) }
        return f
    }

    companion object {
        fun downscale(b: Bitmap, maxSide: Int): Bitmap {
            val s = maxSide.toFloat() / maxOf(b.width, b.height)
            return if (s >= 1f) b else Bitmap.createScaledBitmap(b, (b.width * s).toInt(), (b.height * s).toInt(), true)
        }

        /** Decode with inSampleSize to ≤ ~2048 px and apply EXIF orientation (architecture §4). */
        fun decodeUpright(ctx: android.content.Context, uri: Uri): Bitmap? {
            val cr = ctx.contentResolver
            val o = BitmapFactory.Options().apply { inJustDecodeBounds = true }
            cr.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, o) }
            var ss = 1
            while (maxOf(o.outWidth, o.outHeight) / (ss * 2) >= 1024) ss *= 2
            val bmp = cr.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, BitmapFactory.Options().apply { inSampleSize = ss }) } ?: return null
            val rot = cr.openInputStream(uri)?.use { ExifInterface(it).rotationDegrees } ?: 0
            return if (rot == 0) bmp else Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, Matrix().apply { postRotate(rot.toFloat()) }, true)
        }
    }
}

