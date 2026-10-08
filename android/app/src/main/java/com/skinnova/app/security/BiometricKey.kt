package com.skinnova.app.security

import android.os.Build
import android.security.keystore.KeyGenParameterSpec
import android.security.keystore.KeyPermanentlyInvalidatedException
import android.security.keystore.KeyProperties
import java.security.KeyStore
import javax.crypto.Cipher
import javax.crypto.KeyGenerator
import javax.crypto.SecretKey

/**
 * Fingerprint unlock bound to a Keystore key that needs a strong biometric for every use and is destroyed when the set of
 * enrolled fingerprints changes (setInvalidatedByBiometricEnrollment). Someone who knows the phone's screen lock and adds
 * their own fingerprint therefore cannot open SkinNova with it: the key is gone, and the PIN is needed again.
 */
object BiometricKey {
    private const val ALIAS = "skinnova_bio_v1"
    private fun ks() = KeyStore.getInstance("AndroidKeyStore").apply { load(null) }

    fun create() {
        delete()
        val b = KeyGenParameterSpec.Builder(ALIAS, KeyProperties.PURPOSE_ENCRYPT or KeyProperties.PURPOSE_DECRYPT)
            .setBlockModes(KeyProperties.BLOCK_MODE_GCM).setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_NONE).setKeySize(256)
            .setUserAuthenticationRequired(true).setInvalidatedByBiometricEnrollment(true)
        if (Build.VERSION.SDK_INT >= 30) b.setUserAuthenticationParameters(0, KeyProperties.AUTH_BIOMETRIC_STRONG)
        KeyGenerator.getInstance(KeyProperties.KEY_ALGORITHM_AES, "AndroidKeyStore").apply { init(b.build()) }.generateKey()
    }

    /** A cipher to hand to BiometricPrompt, or null when the key is missing or was invalidated by a new fingerprint. */
    fun cipher(): Cipher? = try {
        val key = (ks().getEntry(ALIAS, null) as? KeyStore.SecretKeyEntry)?.secretKey ?: return null
        Cipher.getInstance("AES/GCM/NoPadding").apply { init(Cipher.ENCRYPT_MODE, key as SecretKey) }
    } catch (e: KeyPermanentlyInvalidatedException) { delete(); null } catch (e: Exception) { null }

    /** True only if the authenticated cipher really works (proves the prompt unlocked this key). */
    fun proves(c: Cipher?): Boolean = runCatching { c != null && c.doFinal(ByteArray(16)).isNotEmpty() }.getOrDefault(false)

    fun delete() { runCatching { ks().deleteEntry(ALIAS) } }
}
