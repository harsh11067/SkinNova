package com.skinnova.app.ml

import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.Bundle
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withContext
import kotlin.coroutines.resume

/**
 * plan §6 USP-2 fallback: when the imported .litertlm has no audio encoder (custom export), transcribe with Android's
 * ON-DEVICE recogniser (API 31+ createOnDeviceSpeechRecognizer; never the cloud one). Extraction stays on Gemma.
 * Hindi needs the phone's offline Hindi speech pack (Settings → Languages → On-device speech recognition).
 */
object OnDeviceSpeech {
    fun available(ctx: Context): Boolean =
        Build.VERSION.SDK_INT >= 33 && SpeechRecognizer.isOnDeviceRecognitionAvailable(ctx)

    /** Must be called on the main thread's looper (SpeechRecognizer requirement). Returns transcript or "" on failure. */
    suspend fun listen(ctx: Context, lang: String, onLevel: (Float) -> Unit): String = withContext(Dispatchers.Main) {
        if (Build.VERSION.SDK_INT < 31) return@withContext ""
        suspendCancellableCoroutine { cont ->
            val sr = SpeechRecognizer.createOnDeviceSpeechRecognizer(ctx)
            cont.invokeOnCancellation { sr.cancel(); sr.destroy() }
            sr.setRecognitionListener(object : RecognitionListener {
                override fun onResults(results: Bundle?) {
                    val t = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)?.firstOrNull() ?: ""
                    sr.destroy(); if (cont.isActive) cont.resume(t)
                }
                override fun onError(error: Int) { sr.destroy(); if (cont.isActive) cont.resume("") }
                override fun onRmsChanged(rmsdB: Float) { onLevel(((rmsdB + 2f) / 12f).coerceIn(0f, 1f)) }
                override fun onReadyForSpeech(params: Bundle?) {}
                override fun onBeginningOfSpeech() {}
                override fun onBufferReceived(buffer: ByteArray?) {}
                override fun onEndOfSpeech() {}
                override fun onPartialResults(partialResults: Bundle?) {}
                override fun onEvent(eventType: Int, params: Bundle?) {}
            })
            val tag = if (lang == "hi") "hi-IN" else "en-IN"
            sr.startListening(Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
                putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
                putExtra(RecognizerIntent.EXTRA_LANGUAGE, tag)
                putExtra(RecognizerIntent.EXTRA_PREFER_OFFLINE, true)
                putExtra(RecognizerIntent.EXTRA_SPEECH_INPUT_COMPLETE_SILENCE_LENGTH_MILLIS, 2500L)
            })
        }
    }
}
