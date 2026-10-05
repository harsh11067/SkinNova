package com.skinnova.app.reminders

import android.Manifest
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import androidx.core.app.NotificationCompat
import androidx.core.app.NotificationManagerCompat
import androidx.core.content.ContextCompat
import androidx.work.CoroutineWorker
import androidx.work.ExistingPeriodicWorkPolicy
import androidx.work.PeriodicWorkRequestBuilder
import androidx.work.WorkManager
import androidx.work.WorkerParameters
import androidx.work.workDataOf
import com.skinnova.app.MainActivity
import com.skinnova.app.R
import java.util.concurrent.TimeUnit

/** "Time to re-check …" — WorkManager periodic job, local notification only (architecture §7). */
class ReminderWorker(ctx: Context, params: WorkerParameters) : CoroutineWorker(ctx, params) {
    override suspend fun doWork(): Result {
        val name = inputData.getString("name") ?: return Result.success()
        notify(applicationContext, inputData.getString("spotId") ?: "", name)
        return Result.success()
    }

    companion object {
        private const val CHANNEL = "spot_reminders"

        fun schedule(ctx: Context, spotId: String, name: String, days: Int) {
            val req = PeriodicWorkRequestBuilder<ReminderWorker>(days.toLong(), TimeUnit.DAYS)
                .setInitialDelay(days.toLong(), TimeUnit.DAYS).addTag("spot_reminder")
                .setInputData(workDataOf("spotId" to spotId, "name" to name)).build()
            WorkManager.getInstance(ctx).enqueueUniquePeriodicWork("reminder_$spotId", ExistingPeriodicWorkPolicy.UPDATE, req)
        }

        fun cancel(ctx: Context, spotId: String) = WorkManager.getInstance(ctx).cancelUniqueWork("reminder_$spotId")

        fun notify(ctx: Context, spotId: String, name: String) {
            val nm = ctx.getSystemService(NotificationManager::class.java)
            nm.createNotificationChannel(NotificationChannel(CHANNEL, "Re-check reminders", NotificationManager.IMPORTANCE_DEFAULT))
            if (ContextCompat.checkSelfPermission(ctx, Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED &&
                android.os.Build.VERSION.SDK_INT >= 33) return
            val pi = PendingIntent.getActivity(ctx, spotId.hashCode(), Intent(ctx, MainActivity::class.java), PendingIntent.FLAG_IMMUTABLE)
            val n = NotificationCompat.Builder(ctx, CHANNEL).setSmallIcon(R.drawable.ic_launcher)
                .setContentTitle(ctx.getString(R.string.tl_reminder_title, name)).setContentText(ctx.getString(R.string.tl_reminder_body))
                .setContentIntent(pi).setAutoCancel(true).build()
            NotificationManagerCompat.from(ctx).notify(spotId.hashCode(), n)
        }
    }
}
