package com.skinnova.app.notify

import android.annotation.SuppressLint
import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.Build
import android.os.PowerManager
import android.provider.Settings

/**
 * Phone makers' power managers stop big background apps: on the test vivo, `com.vivo.abe` force-stopped SkinNova five
 * times in a day, once while the model was writing (exit-info 2026-10-08). Being exempt from battery optimisation (and, on
 * vivo, "allow background power use") keeps an analysis alive when the user switches apps.
 */
object Background {
    fun exempt(ctx: Context): Boolean = ctx.getSystemService(PowerManager::class.java).isIgnoringBatteryOptimizations(ctx.packageName)

    /** The phone's own "allow" dialog; falls back to the battery-optimisation list. */
    @SuppressLint("BatteryLife")
    fun request(ctx: Context) {
        (ctx.applicationContext as? com.skinnova.app.SkinNovaApp)?.container?.lock?.expectExternal()
        val ask = Intent(Settings.ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS, Uri.parse("package:${ctx.packageName}"))
        runCatching { ctx.startActivity(ask) }.onFailure { runCatching { ctx.startActivity(Intent(Settings.ACTION_IGNORE_BATTERY_OPTIMIZATION_SETTINGS)) } }
    }

    /** Makers whose own task killers need an extra, manual "allow" step (shown as text). */
    val aggressiveMaker: Boolean get() = Build.MANUFACTURER.lowercase() in setOf("vivo", "iqoo", "oppo", "realme", "oneplus", "xiaomi", "redmi", "poco", "huawei", "honor")
}
