package com.skinnova.app.i18n

import android.content.Context
import android.speech.tts.TextToSpeech
import android.widget.Toast
import com.skinnova.app.R
import java.util.Locale

/** Android system TTS (offline voices if installed). Missing voice → hint to install it (test.md V9). */
object Tts {
    private var tts: TextToSpeech? = null
    private var ready = false

    fun speak(ctx: Context, text: String, lang: String) {
        val locale = if (lang == "hi") Locale("hi", "IN") else Locale.ENGLISH
        val t = tts
        if (t == null) {
            tts = TextToSpeech(ctx.applicationContext) { status -> ready = status == TextToSpeech.SUCCESS; if (ready) speak(ctx, text, lang) }
            return
        }
        if (!ready) return
        val avail = t.isLanguageAvailable(locale)
        if (avail < TextToSpeech.LANG_AVAILABLE) {
            Toast.makeText(ctx, ctx.getString(R.string.res_tts_missing), Toast.LENGTH_LONG).show(); return
        }
        t.language = locale
        t.speak(text, TextToSpeech.QUEUE_FLUSH, null, "skinnova")
    }

    fun stop() { tts?.stop() }
}
