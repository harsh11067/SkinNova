package com.skinnova.app

import com.skinnova.app.care.Blocked
import com.skinnova.app.care.Caution
import com.skinnova.app.care.ReliefDb
import com.skinnova.app.care.ReliefPlanner
import com.skinnova.app.data.Profile
import com.skinnova.app.ml.CvClassifier
import com.skinnova.app.model.Labels
import com.skinnova.app.model.QuestionnaireAnswers
import com.skinnova.app.model.SnJson
import com.skinnova.app.model.Tier
import com.skinnova.app.security.AppLock
import com.skinnova.app.security.LockStore
import com.skinnova.app.security.Unlock
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

private class MapStore : LockStore {
    val m = HashMap<String, Any>()
    override fun str(k: String) = m[k] as? String
    override fun long(k: String) = when (val v = m[k]) { is Long -> v; is Int -> v.toLong(); is Boolean -> if (v) 1L else 0L; else -> 0L }
    override fun put(vararg kv: Pair<String, Any?>) { kv.forEach { (k, v) -> if (v == null) m.remove(k) else m[k] = v } }
}

/** App lock (security/AppLock.kt): hashing, lock-out, re-lock delay, weak PINs. */
class AppLockTest {
    private var t = 1_000_000L
    private fun lock(s: MapStore = MapStore()) = AppLock(s) { t }

    @Test fun pinIsHashedNotStored() {
        val s = MapStore(); val l = lock(s)
        l.setPin("4826")
        assertTrue(l.enabled)
        assertFalse("PIN must never be stored in clear", s.m.values.any { it.toString().contains("4826") })
        assertEquals(Unlock.Ok, l.check("4826"))
        assertTrue(l.check("4827") is Unlock.Wrong)
    }

    @Test fun lockoutDoublesAndPersists() {
        val s = MapStore(); val l = lock(s); l.setPin("4826")
        repeat(4) { assertTrue(l.check("0000") is Unlock.Wrong) }
        val r = l.check("0000") as Unlock.LockedOut
        assertEquals(t + AppLock.BASE_LOCKOUT, r.untilMs)
        // the right PIN does not work during a lock-out — and a "restart" (new AppLock on the same store) keeps it
        assertTrue(lock(s).check("4826") is Unlock.LockedOut)
        t += AppLock.BASE_LOCKOUT + 1
        assertEquals(t + 2 * AppLock.BASE_LOCKOUT, (l.check("0000") as Unlock.LockedOut).untilMs)
        t += 2 * AppLock.BASE_LOCKOUT + 1
        assertEquals(Unlock.Ok, l.check("4826"))
        assertTrue("counter reset after success", l.check("1111") is Unlock.Wrong)
    }

    @Test fun relocksAfterGraceOnly() {
        val l = lock(); l.setPin("4826"); l.check("4826")
        assertFalse(l.locked.value)
        l.onHidden(); t += AppLock.DEFAULT_GRACE - 1; l.onVisible()
        assertFalse("photo picker / permission dialog within the delay keeps it open", l.locked.value)
        l.onHidden(); t += AppLock.DEFAULT_GRACE; l.onVisible()
        assertTrue(l.locked.value)
        assertTrue("cold start is locked", AppLock(MapStore().also { s -> lock(s).setPin("4826") }) { t }.locked.value)
    }

    @Test fun pinLengthStoredSoPrefixesAreNeverChecked() {
        val s = MapStore(); val l = lock(s); l.setPin("190274")
        assertEquals(6, l.pinLength)
        assertEquals(Unlock.Ok, l.check("190274"))
        l.disable(); assertEquals(0, l.pinLength)
    }

    @Test fun weakPinsRefused() {
        listOf("123", "123456789", "12a4", "1111", "1234", "9876", "0123").forEach { assertNotNull(it, AppLock.validPin(it)) }
        listOf("4826", "190274", "1122").forEach { assertNull(it, AppLock.validPin(it)) }
    }

    @Test fun disableClearsEverything() {
        val s = MapStore(); val l = lock(s); l.setPin("4826"); l.setBiometric(true); l.disable()
        assertFalse(l.enabled); assertFalse(l.locked.value); assertFalse(l.biometric.value)
        assertTrue(s.m.keys.none { it in setOf("hash", "salt", "bio") })
    }
}

/** Home care & relief (assets/care/relief.json + care/Relief.kt). */
class ReliefTest {
    private val db = ReliefDb.parse(File("src/main/assets/care/relief.json").readText())
    private val labels = SnJson.decodeFromString(Labels.serializer(), File("src/main/assets/labels.json").readText())
    private val rx = File("src/main/assets/safety/rx_terms.txt").readLines().map { it.trim().lowercase() }.filter { it.isNotEmpty() && !it.startsWith("#") }
    /** Pharmacy-medicine names the curated notes may use (the LLM still may not: ContentGuards). */
    private val otcAllowed = setOf("hydrocortisone", "clotrimazole", "miconazole", "terbinafine", "ketoconazole", "permethrin", "benzoyl peroxide", "salicylic acid")
    private fun ans(age: String = "18_39", itch: Int = 2, pain: Int = 0) =
        QuestionnaireAnswers("arm", "1_4w", itch, pain, "no", false, false, false, false, age)

    @Test fun everyCategoryHasCuratedCareInBothLanguages() {
        labels.keys.forEach { k -> assertNotNull("relief card for $k", db.cards[k]) }
        val all = db.cards.values.flatMap { c -> c.home.map { it.en to it.hi } + c.food.map { it.en to it.hi } + c.pharmacy.map { it.en to it.hi } } +
            listOf(db.general.itch, db.general.pain).flatMap { g -> g.home.map { it.en to it.hi } + g.pharmacy.map { it.en to it.hi } }
        all.forEach { (en, hi) -> assertTrue(en.isNotBlank() && hi.isNotBlank()); assertTrue("Hindi text expected: $hi", hi.any { it in 'ऀ'..'ॿ' }) }
        db.cards.forEach { (k, c) -> assertTrue("sources for $k", c.sources.isNotEmpty()); assertNotNull("contagious line for $k", c.contagious) }
        assertTrue(db.cards["tinea"]!!.contagious!!.en.startsWith("Contagious")); assertTrue(db.cards["eczema_atopic"]!!.contagious!!.en.startsWith("Not contagious"))
    }

    @Test fun onlyPharmacyMedicinesNamed() {
        val text = File("src/main/assets/care/relief.json").readText().lowercase()
        val named = rx.filter { Regex("(?<![a-z])" + Regex.escape(it) + "(?![a-z])").containsMatchIn(text) }.toSet()
        assertEquals("prescription-only medicine named in relief.json: ${named - otcAllowed}", emptySet<String>(), named - otcAllowed)
        assertFalse("no doses (mg) in curated notes", Regex("""\d\s*mg\b""").containsMatchIn(text))
    }

    @Test fun doctorFirstForHighTierAndSuspiciousSpots() {
        val p = ReliefPlanner.plan(db, "suspicious_lesion", "higher", 0.8, setOf("suspicious_lesion"), Tier.HIGH, ans(), Profile())
        assertEquals(Blocked.DOCTOR_FIRST, p.blocked); assertTrue(p.pharmacy.isEmpty())
        val q = ReliefPlanner.plan(db, "eczema_atopic", "higher", 0.8, setOf("eczema_atopic"), Tier.URGENT, ans(), Profile())
        assertEquals(Blocked.DOCTOR_FIRST, q.blocked)
    }

    @Test fun childrenGetNoMedicines() {
        val p = ReliefPlanner.plan(db, "tinea", "higher", 0.9, setOf("tinea"), Tier.LOW, ans(age = "lt_12"), Profile())
        assertEquals(Blocked.CHILD, p.blocked); assertTrue(p.pharmacy.isEmpty()); assertTrue(p.home.isNotEmpty())
    }

    @Test fun specificTreatmentOnlyWithAClearLead() {
        val sure = ReliefPlanner.plan(db, "tinea", "higher", 0.8, setOf("tinea", "eczema_atopic"), Tier.LOW, ans(), Profile())
        assertTrue(sure.pharmacy.any { it.id == "antifungal_cream" })
        val unsure = ReliefPlanner.plan(db, "tinea", "possible", 0.35, setOf("tinea", "eczema_atopic"), Tier.LOW, ans(), Profile())
        assertFalse(unsure.pharmacy.any { it.specific }); assertTrue(Caution.CONFIRM_FIRST in unsure.cautions)
        assertTrue("itch relief still offered", unsure.pharmacy.any { it.id == "antihistamine" })
    }

    @Test fun noSteroidWhenFungusIsPossible() {
        val p = ReliefPlanner.plan(db, "eczema_atopic", "higher", 0.7, setOf("eczema_atopic", "tinea"), Tier.LOW, ans(), Profile())
        assertFalse(p.pharmacy.any { it.id == "hydrocortisone" })
        val q = ReliefPlanner.plan(db, "eczema_atopic", "higher", 0.7, setOf("eczema_atopic", "psoriasis"), Tier.LOW, ans(), Profile())
        assertTrue(q.pharmacy.any { it.id == "hydrocortisone" })
    }

    @Test fun profileTailorsPharmacyNotes() {
        val pregnant = Profile(sex = "female", pregnant = true, allergies = "lanolin")
        val p = ReliefPlanner.plan(db, "other", "higher", 0.6, setOf("other"), Tier.MODERATE, ans(pain = 2), pregnant)
        assertFalse("no ibuprofen in pregnancy", p.pharmacy.any { it.id == "ibuprofen" })
        assertTrue(p.pharmacy.any { it.id == "paracetamol" })
        assertTrue(Caution.PREGNANT in p.cautions); assertTrue(Caution.ALLERGIES in p.cautions)
        val asthma = ReliefPlanner.plan(db, "other", "higher", 0.6, setOf("other"), Tier.MODERATE, ans(pain = 2), Profile(conditions = listOf("asthma")))
        assertFalse(asthma.pharmacy.any { it.id == "ibuprofen" })
        val diabetic = ReliefPlanner.plan(db, "tinea", "higher", 0.8, setOf("tinea"), Tier.LOW, ans(), Profile(conditions = listOf("diabetes")))
        assertTrue(Caution.INFECTION_RISK in diabetic.cautions)
    }
}

/** TTA views ≡ torch flip(-1) / flip(-2) on NCHW (ml/cv/tta_eval.py). */
class TtaViewTest {
    @Test fun flipsMatchTorch() {
        val n = 3
        val x = FloatArray(n * n * 3) { it.toFloat() }   // value encodes (y, x, c)
        fun at(a: FloatArray, y: Int, xx: Int, c: Int) = a[(y * n + xx) * 3 + c]
        val h = CvClassifier.view(x, n, "h"); val v = CvClassifier.view(x, n, "v"); val r = CvClassifier.view(x, n, "r180")
        for (y in 0 until n) for (xx in 0 until n) for (c in 0 until 3) {
            assertEquals(at(x, y, n - 1 - xx, c), at(h, y, xx, c))
            assertEquals(at(x, n - 1 - y, xx, c), at(v, y, xx, c))
            assertEquals(at(x, n - 1 - y, n - 1 - xx, c), at(r, y, xx, c))
        }
        assertTrue(CvClassifier.view(x, n, "id") === x)
    }
}

/** Ask SkinNova deterministic checks (safety/ChatSafety.kt). */
class ChatSafetyTest {
    private val guards = com.skinnova.app.safety.ContentGuards(
        com.skinnova.app.safety.ContentGuards.parseTerms(File("src/main/assets/safety/rx_terms.txt").readText()))

    @Test fun dangerSignsInEnglishAndHindi() {
        listOf("It is spreading fast and I have a fever", "my lips are swollen", "I can't breathe properly", "rash near my eye",
            "bleeding won't stop", "मुझे बुखार है", "साँस लेने में दिक्कत").forEach { assertTrue(it, com.skinnova.app.safety.ChatSafety.dangerSigns(it)) }
        listOf("Is it contagious?", "What foods should I avoid?", "क्या यह छूत की बीमारी है?").forEach { assertFalse(it, com.skinnova.app.safety.ChatSafety.dangerSigns(it)) }
    }

    @Test fun answersWithMedicineDoseOrDiagnosisAreWithheld() {
        val w = { a: String -> com.skinnova.app.safety.ChatSafety.withheld(a, guards) }
        assertNotNull(w("Choose a cream with clotrimazole or miconazole."))
        assertNotNull(w("Take 500 mg twice a day."))
        assertNotNull(w("You have eczema."))
        assertNotNull(w("{\"possible_categories\": []}"))
        assertNotNull(w(""))
        assertNull(w("If you have a fever, see a doctor today. Eczema is not contagious."))
        assertNull(w("Fungal infections spread through towels; keep the area dry and ask a pharmacist about creams."))
    }

    @Test fun inventedPlacesAndCoinAdviceAreRemoved() {   // real probe answers (reports/chat_probe_v2.json)
        assertEquals("Please show this spot to a doctor or dermatologist soon.",
            com.skinnova.app.safety.ChatSafety.tidy("Please show this spot to a doctor or dermatologist soon. The closest care centre is 15 miles away. "))
        assertEquals("It is more likely spot that needs a doctor's look, so please follow the advice level.",
            com.skinnova.app.safety.ChatSafety.tidy("It is more likely spot that needs a doctor's look, so please follow the advice level. Retake the photo with a coin for size. "))
        assertEquals("Eczema fits best so far. It is not contagious.", com.skinnova.app.safety.ChatSafety.tidy("Eczema fits best so far. It is not contagious."))
        // photo advice only when asked about the photo (phone tour answer, 2026-10-08)
        val a = "It is not contagious. Retake the photo if you change your mind."
        assertEquals("It is not contagious.", com.skinnova.app.safety.ChatSafety.tidy(a, "Is it contagious?"))
        assertEquals(a, com.skinnova.app.safety.ChatSafety.tidy(a, "Should I take a new photo?"))
    }
}
