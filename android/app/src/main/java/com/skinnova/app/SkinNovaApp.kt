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
        // one-time: engine caches written to the cache root by builds before 2026-10-08 (now cache/engine) — up to 3.7 GB
        // (GPU + CPU-fallback weight caches) that nothing reads any more
        Thread {
            cacheDir.listFiles()?.filter { it.isFile && (it.name.contains(".litertlm_") || it.name.endsWith(".xnnpack_cache")) }?.forEach { it.delete() }
        }.start()
        // app lock: re-lock after the app has been out of sight for the chosen delay (security/AppLock.kt)
        androidx.lifecycle.ProcessLifecycleOwner.get().lifecycle.addObserver(object : androidx.lifecycle.DefaultLifecycleObserver {
            override fun onStop(owner: androidx.lifecycle.LifecycleOwner) = container.lock.onHidden()
            override fun onStart(owner: androidx.lifecycle.LifecycleOwner) = container.lock.onVisible()
        })
    }

    /** Free the ~1–2 GB engine when the app is in the background and memory is tight (architecture §4). */
    @Deprecated("Deprecated in Java")
    override fun onTrimMemory(level: Int) {
        super.onTrimMemory(level)
        @Suppress("DEPRECATION")
        if (level >= ComponentCallbacks2.TRIM_MEMORY_BACKGROUND) container.engineHolder.releaseIfIdle()
    }
}
