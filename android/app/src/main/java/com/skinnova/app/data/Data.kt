package com.skinnova.app.data

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.security.keystore.KeyGenParameterSpec
import android.security.keystore.KeyProperties
import androidx.room.Dao
import androidx.room.Database
import androidx.room.Entity
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.PrimaryKey
import androidx.room.Query
import androidx.room.Room
import androidx.room.RoomDatabase
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.withContext
import java.io.ByteArrayOutputStream
import java.io.File
import java.security.KeyStore
import java.util.UUID
import javax.crypto.Cipher
import javax.crypto.KeyGenerator
import javax.crypto.SecretKey
import javax.crypto.spec.GCMParameterSpec

// ---------------- Room (architecture §7, §9) ----------------
@Entity(tableName = "analysis")
data class AnalysisEntity(
    @PrimaryKey val id: String, val createdAt: Long, val imageRef: String, val resultJson: String,
    val topKey: String, val tier: String, val mode: String,
)

@Entity(tableName = "spot")
data class SpotEntity(
    @PrimaryKey val id: String, val name: String, val bodySite: String, val seedX: Float, val seedY: Float,
    val coinDiameterMm: Double?, val reminderDays: Int, val createdAt: Long, val baselineCaptureId: String?,
    val lesionType: Boolean, val noiseArea: Double? = null, val noiseContrast: Double? = null, val noiseN: Int = 0,
)

@Entity(tableName = "capture")
data class CaptureEntity(
    @PrimaryKey val id: String, val spotId: String, val imageRef: String, val takenAt: Long, val cvProbsJson: String,
    val analysisId: String?, val calibration: Boolean = false,
)

@Entity(tableName = "metrics")
data class MetricsEntity(
    @PrimaryKey val captureId: String, val baselineCaptureId: String, val json: String, val timelineTier: String, val narration: String?,
)

@Dao
interface SnDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE) suspend fun put(a: AnalysisEntity)
    @Query("SELECT * FROM analysis ORDER BY createdAt DESC") fun analyses(): Flow<List<AnalysisEntity>>
    @Query("SELECT * FROM analysis WHERE id = :id") suspend fun analysis(id: String): AnalysisEntity?
    @Query("SELECT COUNT(*) FROM analysis") fun analysisCount(): Flow<Int>

    @Insert(onConflict = OnConflictStrategy.REPLACE) suspend fun put(s: SpotEntity)
    @Query("SELECT * FROM spot ORDER BY createdAt DESC") fun spots(): Flow<List<SpotEntity>>
    @Query("SELECT * FROM spot WHERE id = :id") suspend fun spot(id: String): SpotEntity?
    @Query("SELECT * FROM spot") suspend fun allSpots(): List<SpotEntity>
    @Query("DELETE FROM spot WHERE id = :id") suspend fun deleteSpot(id: String)

    @Insert(onConflict = OnConflictStrategy.REPLACE) suspend fun put(c: CaptureEntity)
    @Query("SELECT * FROM capture WHERE spotId = :spotId ORDER BY takenAt") fun captures(spotId: String): Flow<List<CaptureEntity>>
    @Query("SELECT * FROM capture WHERE spotId = :spotId ORDER BY takenAt") suspend fun capturesNow(spotId: String): List<CaptureEntity>
    @Query("SELECT * FROM capture WHERE id = :id") suspend fun capture(id: String): CaptureEntity?
    @Query("DELETE FROM capture WHERE spotId = :spotId") suspend fun deleteCaptures(spotId: String)

    @Insert(onConflict = OnConflictStrategy.REPLACE) suspend fun put(m: MetricsEntity)
    @Query("SELECT * FROM metrics WHERE captureId IN (SELECT id FROM capture WHERE spotId = :spotId)") fun metrics(spotId: String): Flow<List<MetricsEntity>>
    @Query("SELECT * FROM metrics WHERE captureId IN (SELECT id FROM capture WHERE spotId = :spotId)") suspend fun metricsNow(spotId: String): List<MetricsEntity>

    @Query("DELETE FROM analysis") suspend fun clearAnalyses()
    @Query("DELETE FROM spot") suspend fun clearSpots()
    @Query("DELETE FROM capture") suspend fun clearCaptures()
    @Query("DELETE FROM metrics") suspend fun clearMetrics()
}

@Database(entities = [AnalysisEntity::class, SpotEntity::class, CaptureEntity::class, MetricsEntity::class], version = 1, exportSchema = true)
abstract class SnDb : RoomDatabase() {
    abstract fun dao(): SnDao
    companion object {
        fun build(ctx: Context) = Room.databaseBuilder(ctx, SnDb::class.java, "skinnova.db").build()
    }
}

// ---------------- Encrypted image store (architecture §9; test A8) ----------------
/** AES-256-GCM files under filesDir/images; key generated in Android Keystore, non-exportable. File = IV(12) ‖ ciphertext. */
class ImageStore(ctx: Context) {
    private val dir = File(ctx.filesDir, "images").apply { mkdirs() }
    private val alias = "skinnova_images_v1"

    private fun key(): SecretKey {
        val ks = KeyStore.getInstance("AndroidKeyStore").apply { load(null) }
        (ks.getEntry(alias, null) as? KeyStore.SecretKeyEntry)?.let { return it.secretKey }
        val gen = KeyGenerator.getInstance(KeyProperties.KEY_ALGORITHM_AES, "AndroidKeyStore")
        gen.init(KeyGenParameterSpec.Builder(alias, KeyProperties.PURPOSE_ENCRYPT or KeyProperties.PURPOSE_DECRYPT)
            .setBlockModes(KeyProperties.BLOCK_MODE_GCM).setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_NONE)
            .setKeySize(256).build())
        return gen.generateKey()
    }

    suspend fun save(bmp: Bitmap, quality: Int = 90): String = withContext(Dispatchers.IO) {
        val raw = ByteArrayOutputStream().also { bmp.compress(Bitmap.CompressFormat.JPEG, quality, it) }.toByteArray()
        val c = Cipher.getInstance("AES/GCM/NoPadding").apply { init(Cipher.ENCRYPT_MODE, key()) }
        val ref = UUID.randomUUID().toString() + ".enc"
        File(dir, ref).outputStream().use { it.write(c.iv); it.write(c.doFinal(raw)) }
        ref
    }

    suspend fun load(ref: String): Bitmap? = withContext(Dispatchers.IO) {
        val f = File(dir, ref); if (!f.exists()) return@withContext null
        val all = f.readBytes()
        val c = Cipher.getInstance("AES/GCM/NoPadding").apply { init(Cipher.DECRYPT_MODE, key(), GCMParameterSpec(128, all, 0, 12)) }
        val raw = c.doFinal(all, 12, all.size - 12)
        BitmapFactory.decodeByteArray(raw, 0, raw.size)
    }

    fun delete(ref: String) { File(dir, ref).delete() }
    fun deleteAll() { dir.listFiles()?.forEach { it.delete() } }
    fun count() = dir.listFiles()?.size ?: 0
    fun rawFile(ref: String) = File(dir, ref)
}

// ---------------- Settings ----------------
class Settings(ctx: Context) {
    private val p = ctx.getSharedPreferences("settings", Context.MODE_PRIVATE)
    private fun <T> state(v: T) = MutableStateFlow(v)

    private val _dark = state(p.getBoolean("dark", true)); val dark: StateFlow<Boolean> = _dark
    private val _lang = state(p.getString("lang", "en")!!); val lang: StateFlow<String> = _lang
    private val _history = state(p.getBoolean("history", false)); val history: StateFlow<Boolean> = _history
    private val _tts = state(p.getBoolean("tts", true)); val tts: StateFlow<Boolean> = _tts
    private val _onboarded = state(p.getBoolean("onboarded", false)); val onboarded: StateFlow<Boolean> = _onboarded
    private val _setupSeen = state(p.getBoolean("setupSeen", false)); val setupSeen: StateFlow<Boolean> = _setupSeen
    private val _coinMm = state(p.getFloat("coinMm", 20.0f).toDouble()); val coinMm: StateFlow<Double> = _coinMm
    private val _notifAsked = state(p.getBoolean("notifAsked", false)); val notifAsked: StateFlow<Boolean> = _notifAsked
    /** "versionCode|model sha" the GPU cache was last built for (v2.1 item 5: one-time optimising step) */
    private val _optimizedKey = state(p.getString("optimizedKey", "") ?: ""); val optimizedKey: StateFlow<String> = _optimizedKey
    fun setOptimizedKey(v: String) { p.edit().putString("optimizedKey", v).apply(); _optimizedKey.value = v }

    fun setDark(v: Boolean) { p.edit().putBoolean("dark", v).apply(); _dark.value = v }
    fun setLang(v: String) { p.edit().putString("lang", v).apply(); _lang.value = v }
    fun setHistory(v: Boolean) { p.edit().putBoolean("history", v).apply(); _history.value = v }
    fun setTts(v: Boolean) { p.edit().putBoolean("tts", v).apply(); _tts.value = v }
    fun setOnboarded(v: Boolean) { p.edit().putBoolean("onboarded", v).apply(); _onboarded.value = v }
    fun setSetupSeen(v: Boolean) { p.edit().putBoolean("setupSeen", v).apply(); _setupSeen.value = v }
    fun setNotifAsked(v: Boolean) { p.edit().putBoolean("notifAsked", v).apply(); _notifAsked.value = v }
    fun setCoinMm(v: Double) { p.edit().putFloat("coinMm", v.toFloat()).apply(); _coinMm.value = v }
}
