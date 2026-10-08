package com.skinnova.app.data

import android.content.Context
import android.security.keystore.KeyGenParameterSpec
import android.security.keystore.KeyProperties
import com.skinnova.app.model.SnJson
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import java.io.File
import java.security.KeyStore
import javax.crypto.Cipher
import javax.crypto.KeyGenerator
import javax.crypto.SecretKey
import javax.crypto.spec.GCMParameterSpec

/**
 * The user's profile (optional, on this phone only). Used to pre-fill the questionnaire (age band, skin tone) and to
 * tailor the pharmacy notes (children, pregnancy, allergies, conditions that rule out some tablets).
 */
@Serializable
data class Profile(
    val name: String = "",
    @SerialName("age_band") val ageBand: String? = null,
    val sex: String? = null,                                  // female | male | other | null (prefer not to say)
    @SerialName("skin_tone") val skinTone: String = "unknown",
    val pregnant: Boolean = false,                            // pregnant or breastfeeding
    val allergies: String = "",
    val conditions: List<String> = emptyList(),               // Profile.CONDITIONS keys
) {
    val complete: Boolean get() = name.isNotBlank() && ageBand != null && skinTone != "unknown"
    val initial: String get() = name.trim().firstOrNull()?.uppercase() ?: "G"

    companion object {
        /** Conditions that change self-care advice (pharmacy notes / when to see a doctor sooner). */
        val CONDITIONS = listOf("diabetes", "weak_immunity", "asthma", "stomach_ulcer", "kidney_liver")
        val SEXES = listOf("female", "male", "other")
    }
}

/** AES-256-GCM file in app storage; key in Android Keystore (non-exportable), like the history images. */
class ProfileStore(ctx: Context) {
    private val file = File(ctx.filesDir, "profile.enc")
    private val alias = "skinnova_profile_v1"
    private val _profile = MutableStateFlow(runCatching { load() }.getOrNull() ?: Profile()); val profile: StateFlow<Profile> = _profile

    private fun key(): SecretKey {
        val ks = KeyStore.getInstance("AndroidKeyStore").apply { load(null) }
        (ks.getEntry(alias, null) as? KeyStore.SecretKeyEntry)?.let { return it.secretKey }
        val gen = KeyGenerator.getInstance(KeyProperties.KEY_ALGORITHM_AES, "AndroidKeyStore")
        gen.init(KeyGenParameterSpec.Builder(alias, KeyProperties.PURPOSE_ENCRYPT or KeyProperties.PURPOSE_DECRYPT)
            .setBlockModes(KeyProperties.BLOCK_MODE_GCM).setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_NONE).setKeySize(256).build())
        return gen.generateKey()
    }

    private fun load(): Profile? {
        if (!file.exists()) return null
        val all = file.readBytes()
        val c = Cipher.getInstance("AES/GCM/NoPadding").apply { init(Cipher.DECRYPT_MODE, key(), GCMParameterSpec(128, all, 0, 12)) }
        return SnJson.decodeFromString(Profile.serializer(), String(c.doFinal(all, 12, all.size - 12)))
    }

    fun save(p: Profile) {
        val c = Cipher.getInstance("AES/GCM/NoPadding").apply { init(Cipher.ENCRYPT_MODE, key()) }
        val enc = c.doFinal(SnJson.encodeToString(Profile.serializer(), p).toByteArray())
        val tmp = File(file.path + ".tmp")
        tmp.outputStream().use { it.write(c.iv); it.write(enc) }
        tmp.renameTo(file)
        _profile.value = p
    }

    fun delete() { file.delete(); _profile.value = Profile() }
}
