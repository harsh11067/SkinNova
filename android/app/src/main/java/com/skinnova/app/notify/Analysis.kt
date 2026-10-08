package com.skinnova.app.notify

import android.Manifest
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.IBinder
import androidx.core.app.NotificationCompat
import androidx.core.app.NotificationManagerCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.ProcessLifecycleOwner
import com.skinnova.app.MainActivity
import com.skinnova.app.R
import com.skinnova.app.model.Tier

/**
 * "Your result is ready" (local notification only — no network). Lock-screen copy never names a condition or tier
 * (VISIBILITY_PRIVATE + a neutral public version): skin results are health data.
 */
object Notifier {
    private const val CH_PROGRESS = "analysis_progress"
    private const val CH_READY = "analysis_ready"
    const val ID_PROGRESS = 41
    private const val ID_READY = 42

    fun ensureChannels(ctx: Context) {
        val nm = ctx.getSystemService(NotificationManager::class.java)
        nm.createNotificationChannel(NotificationChannel(CH_PROGRESS, ctx.getString(R.string.ntf_ch_progress), NotificationManager.IMPORTANCE_LOW))
        nm.createNotificationChannel(NotificationChannel(CH_READY, ctx.getString(R.string.ntf_ch_ready), NotificationManager.IMPORTANCE_HIGH))
    }

    private fun open(ctx: Context) = PendingIntent.getActivity(ctx, 0,
        Intent(ctx, MainActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP or Intent.FLAG_ACTIVITY_CLEAR_TOP),
        PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT)

    private fun neutral(ctx: Context, ch: String, title: Int) = NotificationCompat.Builder(ctx, ch).setSmallIcon(R.drawable.ic_launcher)
        .setContentTitle(ctx.getString(title)).build()

    fun progress(ctx: Context): Notification = NotificationCompat.Builder(ctx, CH_PROGRESS).setSmallIcon(R.drawable.ic_launcher)
        .setContentTitle(ctx.getString(R.string.ntf_progress_title)).setContentText(ctx.getString(R.string.ntf_progress_body))
        .setProgress(0, 0, true).setOngoing(true).setOnlyAlertOnce(true).setContentIntent(open(ctx))
        .setVisibility(NotificationCompat.VISIBILITY_PRIVATE).setPublicVersion(neutral(ctx, CH_PROGRESS, R.string.ntf_progress_title))
        .setForegroundServiceBehavior(NotificationCompat.FOREGROUND_SERVICE_IMMEDIATE).build()

    private fun allowed(ctx: Context) = Build.VERSION.SDK_INT < 33 ||
        ContextCompat.checkSelfPermission(ctx, Manifest.permission.POST_NOTIFICATIONS) == PackageManager.PERMISSION_GRANTED

    /** Posted when the analysis finishes while SkinNova is not on screen. */
    fun ready(ctx: Context, tier: Tier, basic: Boolean) {
        if (!allowed(ctx)) return
        ensureChannels(ctx)
        val tierText = ctx.getString(when (tier) { Tier.LOW -> R.string.tier_LOW; Tier.MODERATE -> R.string.tier_MODERATE
            Tier.HIGH -> R.string.tier_HIGH; Tier.URGENT -> R.string.tier_URGENT })
        val n = NotificationCompat.Builder(ctx, CH_READY).setSmallIcon(R.drawable.ic_launcher)
            .setContentTitle(ctx.getString(R.string.ntf_ready_title))
            .setContentText(ctx.getString(if (basic) R.string.ntf_ready_body_basic else R.string.ntf_ready_body, tierText))
            .setPriority(NotificationCompat.PRIORITY_HIGH).setCategory(NotificationCompat.CATEGORY_STATUS)
            .setContentIntent(open(ctx)).setAutoCancel(true)
            .setVisibility(NotificationCompat.VISIBILITY_PRIVATE).setPublicVersion(neutral(ctx, CH_READY, R.string.ntf_ready_title)).build()
        runCatching { NotificationManagerCompat.from(ctx).notify(ID_READY, n) }
    }

    fun cancelReady(ctx: Context) = NotificationManagerCompat.from(ctx).cancel(ID_READY)

    /** True while any SkinNova screen is visible (process lifecycle, debounced ~700 ms by androidx). */
    val appVisible: Boolean get() = ProcessLifecycleOwner.get().lifecycle.currentState.isAtLeast(Lifecycle.State.STARTED)
}

/**
 * Holds foreground priority while an analysis runs. The work itself stays in SessionViewModel; this service only keeps
 * the process from being frozen or killed when the user switches apps during the 1–2 min the on-device model needs
 * (the 4 GB engine makes a backgrounded SkinNova the first thing a task killer removes).
 */
class AnalysisService : Service() {
    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        Notifier.ensureChannels(this)
        val n = Notifier.progress(this)
        if (Build.VERSION.SDK_INT >= 34) startForeground(Notifier.ID_PROGRESS, n, ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE)
        else startForeground(Notifier.ID_PROGRESS, n)
        return START_NOT_STICKY
    }

    companion object {
        fun start(ctx: Context) { runCatching { ContextCompat.startForegroundService(ctx, Intent(ctx, AnalysisService::class.java)) } }
        fun stop(ctx: Context) { runCatching { ctx.stopService(Intent(ctx, AnalysisService::class.java)) } }
    }
}
