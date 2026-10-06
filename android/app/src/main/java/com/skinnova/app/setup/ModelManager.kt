package com.skinnova.app.setup

import android.content.Context
import android.net.Uri
import android.os.StatFs
import com.skinnova.app.model.SnJson
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.decodeFromJsonElement
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import java.io.File
import java.io.InputStream
import java.security.MessageDigest
import kotlin.coroutines.coroutineContext

/**
 * Locates / imports / verifies the .litertlm (CLAUDE.md non-negotiable 5: never in the APK, sha256 against
 * assets/model_manifest.json). Sources: (1) adb push → getExternalFilesDir/models, (2) SAF picker → filesDir/models,
 * (3) online flavor download → same as (2).
 */
@Serializable
data class AcceptedModel(val id: String, val label: String, val bytes: Long, val sha256: String)

sealed interface ImportState {
    data object Idle : ImportState
    data class Copying(val fraction: Float) : ImportState
    data object Verifying : ImportState
    data class Ready(val model: AcceptedModel) : ImportState
    data class Error(val kind: Kind, val detail: String = "") : ImportState { enum class Kind { SPACE, WRONG_FILE, IO } }
}

class ModelManager(private val ctx: Context, private val fs: FileOps = FileOps.Real) {
    val fileName: String
    val accepted: List<AcceptedModel>

    init {
        val m = SnJson.parseToJsonElement(ctx.assets.open("model_manifest.json").bufferedReader().readText()).jsonObject["llm"]!!.jsonObject
        fileName = (m["file"] as kotlinx.serialization.json.JsonPrimitive).content
        accepted = m["accepted"]!!.jsonArray.map { SnJson.decodeFromJsonElement<AcceptedModel>(it as JsonObject) }.filter { it.sha256.isNotEmpty() }
    }

    private val prefs = ctx.getSharedPreferences("model", Context.MODE_PRIVATE)
    val internalDir: File get() = File(ctx.filesDir, "models").apply { mkdirs() }
    val externalDir: File? get() = ctx.getExternalFilesDir("models")?.apply { mkdirs() }

    /** Largest expected file + 500 MB headroom (architecture §10). */
    val requiredBytes: Long get() = (accepted.maxOfOrNull { it.bytes } ?: 2_600_000_000L) + 500_000_000L

    fun freeBytes(): Long = StatFs(ctx.filesDir.path).availableBytes

    /** Verified model path, or null. Verification result is cached by (path, size, mtime) so we hash 2.6 GB only once. */
    fun activeModelPath(): String? = candidates().firstOrNull { verifiedCached(it) != null }?.path

    fun activeModel(): AcceptedModel? = candidates().firstNotNullOfOrNull { verifiedCached(it) }

    private fun candidates(): List<File> = buildList {
        add(File(internalDir, fileName))
        externalDir?.let { d -> d.listFiles { f -> f.name.endsWith(".litertlm") }?.sortedBy { it.name }?.let { addAll(it) } }
    }.filter { it.isFile && it.length() > 0 }

    private fun key(f: File) = "${f.path}|${f.length()}|${f.lastModified()}"

    private fun verifiedCached(f: File): AcceptedModel? {
        val id = prefs.getString(key(f), null) ?: return null
        return accepted.firstOrNull { it.id == id }
    }

    /** Hash any un-verified candidates (adb-pushed files). Call off the main thread. */
    suspend fun scanAndVerify(onProgress: (Float) -> Unit = {}): AcceptedModel? = withContext(Dispatchers.IO) {
        for (f in candidates()) {
            verifiedCached(f)?.let { return@withContext it }
            // several accepted models can share a size (LoRA v1 and v2: identical tensor shapes): hash once, match by sha
            if (accepted.none { it.bytes == f.length() }) continue
            val sha = fs.sha256(f.inputStream(), f.length(), onProgress)
            val match = accepted.firstOrNull { it.bytes == f.length() && it.sha256 == sha } ?: continue
            prefs.edit().putString(key(f), match.id).apply(); return@withContext match
        }
        null
    }

    /** SAF import: space check → copy to .part with progress → sha256 → rename. Cancellation deletes the partial file. */
    suspend fun import(uri: Uri, onState: (ImportState) -> Unit): ImportState = withContext(Dispatchers.IO) {
        val size = ctx.contentResolver.openAssetFileDescriptor(uri, "r")?.use { it.length } ?: -1L
        val need = (if (size > 0) size else requiredBytes - 500_000_000L) + 500_000_000L
        if (freeBytes() < need) return@withContext ImportState.Error(ImportState.Error.Kind.SPACE, need.toString()).also(onState)
        val part = File(internalDir, "$fileName.part")
        val target = File(internalDir, fileName)
        val digest = MessageDigest.getInstance("SHA-256")
        try {
            ctx.contentResolver.openInputStream(uri)!!.use { input ->
                part.outputStream().use { out ->
                    val buf = ByteArray(1 shl 20); var done = 0L; var lastReport = 0L
                    while (true) {
                        coroutineContext.ensureActive()
                        val n = input.read(buf); if (n < 0) break
                        out.write(buf, 0, n); digest.update(buf, 0, n); done += n
                        if (size > 0 && done - lastReport > 16_000_000) { lastReport = done; onState(ImportState.Copying(done.toFloat() / size)) }
                    }
                }
            }
            onState(ImportState.Verifying)
            val sha = digest.digest().joinToString("") { "%02x".format(it) }
            val match = accepted.firstOrNull { it.sha256 == sha }
            if (match == null) { part.delete(); return@withContext ImportState.Error(ImportState.Error.Kind.WRONG_FILE, sha.take(12)).also(onState) }
            target.delete(); check(part.renameTo(target))
            prefs.edit().putString(key(target), match.id).apply()
            ImportState.Ready(match).also(onState)
        } catch (e: kotlinx.coroutines.CancellationException) {
            part.delete(); throw e
        } catch (e: Exception) {
            part.delete(); ImportState.Error(ImportState.Error.Kind.IO, e.message ?: e.javaClass.simpleName).also(onState)
        }
    }

    /** Record a file whose sha256 was already checked while it was written (import / download). */
    fun markVerified(f: File, id: String) { prefs.edit().putString(key(f), id).apply() }

    fun deleteModels() {
        candidates().forEach { it.delete() }
        File(internalDir, "$fileName.part").delete()
        prefs.edit().clear().apply()
    }
}

/** Seam for unit tests (fake FS). */
interface FileOps {
    fun sha256(input: InputStream, size: Long, onProgress: (Float) -> Unit): String

    object Real : FileOps {
        override fun sha256(input: InputStream, size: Long, onProgress: (Float) -> Unit): String = input.use {
            val d = MessageDigest.getInstance("SHA-256"); val buf = ByteArray(1 shl 20); var done = 0L
            while (true) { val n = it.read(buf); if (n < 0) break; d.update(buf, 0, n); done += n; if (size > 0) onProgress(done.toFloat() / size) }
            d.digest().joinToString("") { b -> "%02x".format(b) }
        }
    }
}
