package com.skinnova.app

import android.app.Application
import android.content.ComponentCallbacks2
import com.skinnova.app.di.AppContainer

class SkinNovaApp : Application() {
    lateinit var container: AppContainer
        private set

    override fun onCreate() {
        super.onCreate()
        container = AppContainer(this)
    }

    /** Free the ~1–2 GB engine when the app is in the background and memory is tight (architecture §4). */
    @Deprecated("Deprecated in Java")
    override fun onTrimMemory(level: Int) {
        super.onTrimMemory(level)
        @Suppress("DEPRECATION")
        if (level >= ComponentCallbacks2.TRIM_MEMORY_BACKGROUND) container.engineHolder.release()
    }
}
