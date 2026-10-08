package com.skinnova.app.security

import android.content.Context
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import java.security.MessageDigest
import java.security.SecureRandom
import java.util.Base64
import javax.crypto.SecretKeyFactory
import javax.crypto.spec.PBEKeySpec

/** Key–value storage behind [AppLock] (SharedPreferences on the phone, a map in JVM tests). */
interface LockStore {
    fun str(k: String): String?
    fun long(k: String): Long
    fun put(vararg kv: Pair<String, Any?>)   // null value = remove
}

class PrefsLockStore(ctx: Context) : LockStore {
    private val p = ctx.getSharedPreferences("app_lock", Context.MODE_PRIVATE)
    override fun str(k: String) = p.getString(k, null)
    override fun long(k: String) = p.getLong(k, 0L)
    override fun put(vararg kv: Pair<String, Any?>) {
        val e = p.edit()
        kv.forEach { (k, v) -> when (v) { null -> e.remove(k); is String -> e.putString(k, v); is Long -> e.putLong(k, v); is Int -> e.putLong(k, v.toLong())
            is Boolean -> e.putLong(k, if (v) 1 else 0); else -> error("type") } }
        e.commit()   // synchronous: a crash right after a wrong PIN must not reset the attempt counter
    }
}

sealed interface Unlock {
    data object Ok : Unlock
    data class Wrong(val triesLeft: Int) : Unlock
    data class LockedOut(val untilMs: Long) : Unlock
}

/**
 * App lock: a 4–8 digit PIN (+ optional fingerprint). Everything SkinNova shows is guarded by it once it is set
 * (MainActivity renders no route while [locked]).
 *
 * - PIN never stored: PBKDF2-HMAC-SHA256, 120 000 iterations, 16-byte random salt; constant-time compare.
 * - Wrong PINs: [FREE_TRIES] free tries, then a lock-out that doubles (30 s, 1 min, 2 min … max 15 min), persisted
 *   (a restart does not reset it).
 * - Re-locks when the app has been in the background for [graceMs] (default 1 min, so the photo picker or the
 *   permission dialog do not trigger it); always locked on a cold start.
 */
class AppLock(private val s: LockStore, private val now: () -> Long = System::currentTimeMillis) {
    val enabled: Boolean get() = s.str("hash") != null
    private val _on = MutableStateFlow(enabled); val on: StateFlow<Boolean> = _on   // drives FLAG_SECURE in MainActivity
    private val _locked = MutableStateFlow(enabled); val locked: StateFlow<Boolean> = _locked
    private val _biometric = MutableStateFlow(s.long("bio") == 1L); val biometric: StateFlow<Boolean> = _biometric
    private val _graceMs = MutableStateFlow(s.long("grace").takeIf { it > 0 } ?: DEFAULT_GRACE); val graceMs: StateFlow<Long> = _graceMs
    private var hiddenAt = 0L

    fun setPin(pin: String) {
        require(validPin(pin) == null) { "weak pin" }
        val salt = ByteArray(16).also { SecureRandom().nextBytes(it) }
        s.put("salt" to b64(salt), "hash" to b64(hash(pin, salt, ITERATIONS)), "iter" to ITERATIONS.toLong(), "fails" to null, "until" to null)
        _locked.value = false; _on.value = true
    }

    fun disable() { s.put("salt" to null, "hash" to null, "iter" to null, "fails" to null, "until" to null, "bio" to null); _biometric.value = false; _locked.value = false; _on.value = false }

    fun setBiometric(on: Boolean) { s.put("bio" to on); _biometric.value = on }
    fun setGrace(ms: Long) { s.put("grace" to ms); _graceMs.value = ms }

    fun lockedOutUntil(): Long = s.long("until").takeIf { it > now() } ?: 0L

    fun check(pin: String): Unlock {
        lockedOutUntil().takeIf { it > 0 }?.let { return Unlock.LockedOut(it) }
        val salt = s.str("salt")?.let(::unb64) ?: return Unlock.Ok
        val want = unb64(s.str("hash")!!)
        val got = hash(pin, salt, s.long("iter").toInt().takeIf { it > 0 } ?: ITERATIONS)
        if (MessageDigest.isEqual(want, got)) { s.put("fails" to null, "until" to null); _locked.value = false; return Unlock.Ok }
        val fails = s.long("fails") + 1
        return if (fails >= FREE_TRIES) {
            val wait = minOf(MAX_LOCKOUT, BASE_LOCKOUT shl (fails - FREE_TRIES).toInt().coerceAtMost(10))
            val until = now() + wait
            s.put("fails" to fails, "until" to until); Unlock.LockedOut(until)
        } else { s.put("fails" to fails); Unlock.Wrong((FREE_TRIES - fails).toInt()) }
    }

    /** Called after a successful BiometricPrompt (BIOMETRIC_STRONG). */
    fun unlockWithBiometric() { if (lockedOutUntil() == 0L) { s.put("fails" to null); _locked.value = false } }

    fun lockNow() { if (enabled) _locked.value = true }
    fun onHidden() { hiddenAt = now() }
    fun onVisible() { if (enabled && hiddenAt > 0 && now() - hiddenAt >= _graceMs.value) _locked.value = true; hiddenAt = 0 }

    companion object {
        const val ITERATIONS = 120_000
        const val FREE_TRIES = 5L
        const val BASE_LOCKOUT = 30_000L
        const val MAX_LOCKOUT = 15 * 60_000L
        const val DEFAULT_GRACE = 60_000L

        /** null = acceptable; otherwise a reason key: "length", "digits", "simple" (1111, 1234, 9876…). */
        fun validPin(pin: String): String? = when {
            pin.length !in 4..8 -> "length"
            !pin.all { it in '0'..'9' } -> "digits"
            pin.toSet().size == 1 -> "simple"
            pin.zipWithNext { a, b -> b - a }.toSet().let { it == setOf(1) || it == setOf(-1) } -> "simple"
            else -> null
        }

        fun hash(pin: String, salt: ByteArray, iterations: Int): ByteArray =
            SecretKeyFactory.getInstance("PBKDF2WithHmacSHA256").generateSecret(PBEKeySpec(pin.toCharArray(), salt, iterations, 256)).encoded

        private fun b64(b: ByteArray) = Base64.getEncoder().encodeToString(b)
        private fun unb64(s: String) = Base64.getDecoder().decode(s)
    }
}
