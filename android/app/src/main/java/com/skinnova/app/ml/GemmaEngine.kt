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
        val e = if (!forceCpu) {
            try { build(path, gpu = true).also { backendName = "GPU" } } catch (t: Throwable) {
                Log.w(TAG, "GPU init failed, falling back to CPU", t); build(path, gpu = false).also { backendName = "CPU" }
            }
        } else build(path, gpu = false).also { backendName = "CPU" }
        prefs.edit().putBoolean("lastLoadCrashed", false).putString("backend", backendName).apply()
        initCount++
        Log.i(TAG, "Engine init #$initCount backend=$backendName vision=$supportsVision audio=$supportsAudio")
        return e
    }

    private fun build(path: String, gpu: Boolean): Engine {
        val b = if (gpu) Backend.GPU() else Backend.CPU()
        return Engine(EngineConfig(
            modelPath = path, backend = b,
            visionBackend = if (supportsVision) (if (gpu) Backend.GPU() else Backend.CPU()) else null,
            audioBackend = if (supportsAudio) Backend.CPU() else null,
            maxNumTokens = 4096, cacheDir = ctx.cacheDir.path,
        )).apply { initialize() }
    }

    fun release() { engine?.close(); engine = null; enginePath = null }

    /** Debug/test seam: force the CPU path (test A9). */
    fun forceBackend(name: String) { prefs.edit().putString("backend", name).apply(); backendName = name; release() }

    companion object { const val TAG = "EngineHolder" }
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
    ANALYZE(0.2, 700), REPAIR(0.2, 700), TRANSCRIBE(0.0, 400), EXTRACT(0.2, 300), NARRATE(0.4, 200), TRANSLATE(0.4, 600)
}

class GemmaEngine(private val holder: EngineHolder, private val models: ModelManager) : Llm {
    override val available get() = models.activeModelPath() != null

    override suspend fun generate(task: LlmTask, system: String, user: String, imagePath: String?, audio: ByteArray?,
                                  onToken: (String) -> Unit): String = withTimeout(180_000) {
        holder.use { engine ->
            val conv = engine.createConversation(ConversationConfig(
                systemInstruction = Contents.of(system),
                samplerConfig = SamplerConfig(topK = 40, topP = 0.95, temperature = task.temperature, seed = 3407),
                maxOutputToken = task.maxTokens,
                thinkingConfig = ThinkingConfig(enableThinking = false),
            ))
            try {
                val parts = buildList<Content> {
                    if (imagePath != null && holder.supportsVision) add(Content.ImageFile(imagePath))
                    if (audio != null) add(Content.AudioBytes(audio))
                    add(Content.Text(user))
                }
                val sb = StringBuilder()
                suspendCancellableCoroutine { cont ->
                    cont.invokeOnCancellation { runCatching { conv.cancelProcess() } }
                    conv.sendMessageAsync(Contents.of(parts), object : MessageCallback {
                        override fun onMessage(message: Message) { val t = message.toString(); sb.append(t); onToken(sb.toString()) }
                        override fun onDone() { if (cont.isActive) cont.resume(Unit) }
                        override fun onError(throwable: Throwable) { if (cont.isActive) cont.resumeWithException(throwable) }
                    }, emptyMap())
                }
                sb.toString()
            } finally {
                conv.close()
            }
        }
    }
}
