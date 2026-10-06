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
    var savedId: String? = null; private set
    private var job: Job? = null
    private val audio = com.skinnova.app.ml.AudioCapture()

    // ---------- photo ----------
    fun setPhoto(bmp: Bitmap) {
        val b = downscale(bmp, 1024)
        _photo.value = b; qualityForced = false; _result.value = null; savedId = null; _draft.value = Draft()
        viewModelScope.launch(Dispatchers.Default) { _quality.value = QualityGate.check(b.toRgb()) }
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
        job = viewModelScope.launch {
            try {
                _state.value = AnalysisState.CheckingPhoto
                val rgb = withContext(Dispatchers.Default) { bmp.toRgb() }
                if (_quality.value == null) _quality.value = withContext(Dispatchers.Default) { QualityGate.check(rgb) }
                _state.value = AnalysisState.Classifying
                if (!c.cv.available) { _state.value = AnalysisState.Failed("Image model not found in this build"); return@launch }
                val cv = withContext(Dispatchers.Default) { c.cv.classify(rgb) }
                _cv.value = cv
                val img = withContext(Dispatchers.IO) { writeLlmImage(rgb) }
                val res = c.pipeline().run(answers, cv, qualityForced, img.path, c.settings.lang.value) { _state.value = it }
                _result.value = res
                if (c.settings.history.value) save(res)
                _state.value = AnalysisState.Done(res)
            } catch (e: kotlinx.coroutines.CancellationException) {
                _state.value = AnalysisState.Idle; throw e
            } catch (e: Throwable) {
                _state.value = AnalysisState.Failed(e.message ?: e.javaClass.simpleName)
            }
        }
    }

    fun cancel() { job?.cancel(); _state.value = AnalysisState.Idle }

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
        _result.value = SnJson.decodeFromString(FinalResult.serializer(), e.resultJson)
        _photo.value = c.images.load(e.imageRef); savedId = e.id
        _cv.value = _result.value!!.cvTop3
    }

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

