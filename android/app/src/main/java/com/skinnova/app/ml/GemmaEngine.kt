package com.skinnova.app.ml

import android.content.Context
import android.util.Log
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Capabilities
import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.Message
import com.google.ai.edge.litertlm.MessageCallback
import com.google.ai.edge.litertlm.SamplerConfig
import com.google.ai.edge.litertlm.ThinkingConfig
import com.skinnova.app.setup.ModelManager
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeout
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException

class ModelMissing : Exception("model not imported")

/**
 * One LiteRT-LM Engine per process (architecture §4). GPU first; on init failure CPU, persisted.
 * `lastLoadCrashed` flag survives a process kill during load (OOM) → next start goes CPU.
 */
class EngineHolder(private val ctx: Context, private val models: ModelManager) {
    private val mutex = Mutex()
    private var engine: Engine? = null
    private var enginePath: String? = null
    private val prefs = ctx.getSharedPreferences("engine", Context.MODE_PRIVATE)
    @OptIn(ExperimentalCoroutinesApi::class)
    val llmDispatcher: CoroutineDispatcher = Dispatchers.Default.limitedParallelism(1)
    var backendName: String = prefs.getString("backend", "GPU")!!; private set
    var initCount = 0; private set
    /** Wall time of the last engine load (incl. first-run XNNPack/GPU cache builds) — bench / S8 reporting. */
    var lastInitMs = 0L; private set
    var supportsVision = true; private set
    var supportsAudio = true; private set

    val isLoaded get() = engine != null

    suspend fun <T> use(block: suspend (Engine) -> T): T = withContext(llmDispatcher) {
        mutex.withLock {
            val path = models.activeModelPath() ?: throw ModelMissing()
            val e = engine?.takeIf { enginePath == path } ?: create(path).also { engine = it; enginePath = path }
            block(e)
        }
    }

    private fun create(path: String): Engine {
        engine?.close(); engine = null
        try {
            Capabilities(path).use { c ->
                val m = c.inputModalities(); supportsVision = m.vision; supportsAudio = m.audio
            }
        } catch (t: Throwable) { Log.w(TAG, "capabilities: ${t.message}") }
        val forceCpu = prefs.getBoolean("lastLoadCrashed", false) || backendName == "CPU"
        prefs.edit().putBoolean("lastLoadCrashed", true).commit()   // cleared only if init returns
        val t0 = System.nanoTime()
        val e = if (!forceCpu) {
            try { build(path, gpu = true).also { backendName = "GPU" } } catch (t: Throwable) {
                Log.w(TAG, "GPU init failed, falling back to CPU", t); build(path, gpu = false).also { backendName = "CPU" }
            }
        } else build(path, gpu = false).also { backendName = "CPU" }
        prefs.edit().putBoolean("lastLoadCrashed", false).putString("backend", backendName).apply()
        initCount++; lastInitMs = (System.nanoTime() - t0) / 1_000_000
        Log.i(TAG, "Engine init #$initCount backend=$backendName vision=$supportsVision audio=$supportsAudio in ${lastInitMs} ms")
        return e
    }

    private fun build(path: String, gpu: Boolean): Engine {
        val b = if (gpu) Backend.GPU() else Backend.CPU()
        return Engine(EngineConfig(
            modelPath = path, backend = b,
            // no vision encoder: the photo is not sent (SEND_PHOTO_TO_LLM). When it was, it ran on the CPU — on a Mali GPU
            // it was one 13–17 s job that froze the UI ("not responding", vivo V2059 / Helio G95, 2026-10-07).
            visionBackend = if (SEND_PHOTO_TO_LLM && supportsVision) Backend.CPU() else null,
            audioBackend = if (supportsAudio) Backend.CPU() else null,
            maxNumTokens = 4096, cacheDir = ctx.cacheDir.path,
        )).apply { initialize() }
    }

    fun release() { engine?.close(); engine = null; enginePath = null }

    /** Memory-pressure release: never while a generation holds the engine (closing it mid-run crashed the native side). */
    fun releaseIfIdle() { if (mutex.tryLock()) try { release() } finally { mutex.unlock() } }

    /** Debug/test seam: force the CPU path (test A9). */
    fun forceBackend(name: String) { prefs.edit().putString("backend", name).apply(); backendName = name; release() }

    companion object {
        const val TAG = "EngineHolder"
        /** decisions.md 2026-10-07 (pre-registered): the fine-tuned model's answers are identical without the photo on
         *  the frozen llm_val (category agreement 0.91 = 0.91, JSON valid 1.0 = 1.0) and 46 % faster — the image model
         *  carries the visual evidence, the LLM works from its scores and the answers. */
        const val SEND_PHOTO_TO_LLM = false
    }
}

/**
 * The five LLM tasks (architecture §4): one short-lived conversation per task, own system prompt, closed after use.
 * Interface so the pipeline can be tested with a fake (test.md §10 A1).
 */
interface Llm {
    val available: Boolean
    suspend fun generate(task: LlmTask, system: String, user: String, imagePath: String? = null, audio: ByteArray? = null,
                         onToken: (String) -> Unit = {}): String
}

enum class LlmTask(val temperature: Double, val maxTokens: Int) {
    ANALYZE(0.2, 700), REPAIR(0.2, 700), TRANSCRIBE(0.0, 400), EXTRACT(0.2, 300), NARRATE(0.4, 200), TRANSLATE(0.4, 600), CHAT(0.3, 300)
}

class GemmaEngine(private val holder: EngineHolder, private val models: ModelManager) : Llm {
    override val available get() = models.activeModelPath() != null
    /** Generation limit — the engine load / wait is NOT counted (a mid-range phone needs ~2 min just to load: on-device
     *  2026-10-07, Helio G95 GPU 115 s). Past it the pipeline falls back to Basic mode. Bench raises it to time S8. */
    var timeoutMs = 180_000L

    override suspend fun generate(task: LlmTask, system: String, user: String, imagePath: String?, audio: ByteArray?,
                                  onToken: (String) -> Unit): String = holder.use { engine ->
        onToken("")   // engine ready, generation starting (the UI leaves "safety checks"; the first real token can be ~1 min away)
        val t0 = System.nanoTime(); var firstToken = 0L; var lastUi = 0L
        Log.i(EngineHolder.TAG, "generate ${task.name} start backend=${holder.backendName} image=${imagePath != null && holder.supportsVision}")
        withTimeout(timeoutMs) {
            val conv = engine.createConversation(ConversationConfig(
                systemInstruction = Contents.of(system),
                samplerConfig = SamplerConfig(topK = 40, topP = 0.95, temperature = task.temperature, seed = 3407),
                maxOutputToken = task.maxTokens,
                thinkingConfig = ThinkingConfig(enableThinking = false),
            ))
            try {
                val parts = buildList<Content> {
                    if (imagePath != null && holder.supportsVision && EngineHolder.SEND_PHOTO_TO_LLM) add(Content.ImageFile(imagePath))
                    if (audio != null) add(Content.AudioBytes(audio))
                    add(Content.Text(user))
                }
                val sb = StringBuilder()
                suspendCancellableCoroutine { cont ->
                    cont.invokeOnCancellation { runCatching { conv.cancelProcess() } }
                    conv.sendMessageAsync(Contents.of(parts), object : MessageCallback {
                        override fun onMessage(message: Message) {
                            sb.append(message.toString()); val now = System.nanoTime()
                            if (firstToken == 0L) { firstToken = now; Log.i(EngineHolder.TAG, "generate ${task.name} first token after ${(now - t0) / 1_000_000} ms") }
                            // ≤ ~3 UI updates/s: one recomposition per token competed with the GPU for frames
                            if (now - lastUi > 300_000_000L) { lastUi = now; onToken(sb.toString()) }
                        }
                        override fun onDone() { if (cont.isActive) cont.resume(Unit) }
                        override fun onError(throwable: Throwable) { if (cont.isActive) cont.resumeWithException(throwable) }
                    }, emptyMap())
                }
                Log.i(EngineHolder.TAG, "generate ${task.name} done in ${(System.nanoTime() - t0) / 1_000_000} ms, ${sb.length} chars")
                onToken(sb.toString())
                sb.toString()
            } finally {
                conv.close()
            }
        }
    }
}
