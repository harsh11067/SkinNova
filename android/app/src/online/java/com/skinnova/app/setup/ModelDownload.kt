package com.skinnova.app.setup

import android.content.Context
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import com.skinnova.app.model.SnJson
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import java.io.File
import java.io.RandomAccessFile
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest
import kotlin.coroutines.coroutineContext

/**
 * online flavor only: one-time, resumable (HTTP Range) download on an unmetered network, then the same sha256 check
 * as a file import. URL comes from assets/model_manifest.json ("url" of the first accepted model that has one).
 */
object ModelDownload {
    const val available = true

    suspend fun download(ctx: Context, models: ModelManager, onState: (ImportState) -> Unit): ImportState = withContext(Dispatchers.IO) {
        val cm = ctx.getSystemService(ConnectivityManager::class.java)
        val caps = cm.getNetworkCapabilities(cm.activeNetwork)
        if (caps == null || !caps.hasCapability(NetworkCapabilities.NET_CAPABILITY_NOT_METERED))
            return@withContext ImportState.Error(ImportState.Error.Kind.IO, "connect to Wi-Fi first").also(onState)
        val manifest = SnJson.parseToJsonElement(ctx.assets.open("model_manifest.json").bufferedReader().readText()).jsonObject
        val entry = manifest["llm"]!!.jsonObject["accepted"]!!.jsonArray.map { it.jsonObject }
            .firstOrNull { (it["url"] as? JsonPrimitive)?.content?.isNotBlank() == true }
            ?: return@withContext ImportState.Error(ImportState.Error.Kind.IO, "no download URL in manifest").also(onState)
        val url = (entry["url"] as JsonPrimitive).content
        val sha = (entry["sha256"] as JsonPrimitive).content
        val bytes = (entry["bytes"] as JsonPrimitive).content.toLong()
        if (models.freeBytes() < bytes + 500_000_000L) return@withContext ImportState.Error(ImportState.Error.Kind.SPACE).also(onState)
        val part = File(models.internalDir, models.fileName + ".part")
        try {
            var have = if (part.exists()) part.length() else 0L
            while (have < bytes) {
                val conn = (URL(url).openConnection() as HttpURLConnection).apply {
                    setRequestProperty("Range", "bytes=$have-"); connectTimeout = 15_000; readTimeout = 30_000; instanceFollowRedirects = true
                }
                conn.inputStream.use { input ->
                    RandomAccessFile(part, "rw").use { out ->
                        out.seek(have); val buf = ByteArray(1 shl 20)
                        while (true) {
                            coroutineContext.ensureActive()
                            val n = input.read(buf); if (n < 0) break
                            out.write(buf, 0, n); have += n
                            onState(ImportState.Copying(have.toFloat() / bytes))
                        }
                    }
                }
            }
            onState(ImportState.Verifying)
            val got = part.inputStream().use { s ->
                val d = MessageDigest.getInstance("SHA-256"); val b = ByteArray(1 shl 20)
                while (true) { val n = s.read(b); if (n < 0) break; d.update(b, 0, n) }
                d.digest().joinToString("") { "%02x".format(it) }
            }
            if (got != sha) { part.delete(); return@withContext ImportState.Error(ImportState.Error.Kind.WRONG_FILE, got.take(12)).also(onState) }
            val target = File(models.internalDir, models.fileName); target.delete(); check(part.renameTo(target))
            val m = models.accepted.first { it.sha256 == sha }
            models.markVerified(target, m.id)
            ImportState.Ready(m).also(onState)
        } catch (e: kotlinx.coroutines.CancellationException) {
            throw e   // keep the .part file: the next attempt resumes
        } catch (e: Exception) {
            ImportState.Error(ImportState.Error.Kind.IO, e.message ?: "network error").also(onState)
        }
    }

}
