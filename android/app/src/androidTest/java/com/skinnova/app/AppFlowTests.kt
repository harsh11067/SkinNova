package com.skinnova.app

import android.Manifest
import android.app.Application
import android.graphics.BitmapFactory
import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.hasClickAction
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.junit4.createEmptyComposeRule
import androidx.compose.ui.test.onAllNodesWithText
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performScrollTo
import androidx.compose.ui.semantics.SemanticsProperties
import androidx.compose.ui.semantics.getOrNull
import androidx.compose.ui.unit.Density
import androidx.test.core.app.ActivityScenario
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.rule.GrantPermissionRule
import com.skinnova.app.ml.AnalysisState
import com.skinnova.app.ml.Llm
import com.skinnova.app.ml.LlmTask
import com.skinnova.app.model.Mode
import com.skinnova.app.model.Tier
import com.skinnova.app.ui.Draft
import com.skinnova.app.ui.SessionViewModel
import com.skinnova.app.ui.screens.ResultScreen
import com.skinnova.app.ui.screens.deleteEverything
import com.skinnova.app.ui.theme.SkinNovaTheme
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith

/** A validator-valid analysis (a real SFT target: eczema first, tier LOW) — the rules, not the LLM, must raise the tier. */
private const val FAKE_ANALYSIS = """{"possible_categories":[{"key":"eczema_atopic","likelihood":"higher","why":"Supported by moderate itching; commonly on brown skin may look dark brown, purple or grey rather than red."},{"key":"psoriasis","likelihood":"less_likely","why":"Less likely: psoriasis plaques are thicker with silvery scale and sharp borders."},{"key":"vitiligo","likelihood":"less_likely","why":"Less likely: your answers fit it less well."}],"uncertainty":{"level":"low","reasons":[]},"explanation":"Based on your answers, it has been present for one to six months on the foot, with moderate itching. Eczema / atopic dermatitis fits best so far. A long-lasting tendency to dry, itchy, inflamed skin that comes and goes in flares.","what_would_help":["A doctor's examination in person"],"self_care_info":["Short lukewarm baths and gentle soap-free wash.","Moisturise often with a plain, fragrance-free cream."],"triage":{"tier":"LOW","advice":"Watch it and use gentle care. See a doctor if it spreads, hurts, or doesn't improve in 2 weeks."},"disagreement_with_image_model":false}"""

/** test.md §10 A1: the pipeline with a fake LLM, no model file needed. */
class FakeLlm(private val reply: String) : Llm {
    var calls = 0; private set
    override val available = true
    override suspend fun generate(task: LlmTask, system: String, user: String, imagePath: String?, audio: ByteArray?,
                                  onToken: (String) -> Unit): String { calls++; onToken(reply); return reply }
}

private val app get() = ApplicationProvider.getApplicationContext<Application>() as SkinNovaApp
private fun s(id: Int) = app.getString(id)
private val testAssets get() = InstrumentationRegistry.getInstrumentation().context.assets

/** A child (lt_12) with eczema-like answers: R5 must fire and lift the LLM's LOW to ≥ MODERATE. */
private val CHILD_ECZEMA = Draft("foot", "1_6m", 2, 0, "no", false, false, false, false, "lt_12")

private fun analyzeWithFake(vm: SessionViewModel) = runBlocking {
    val bmp = testAssets.open("cv_fixtures/00.jpg").use { BitmapFactory.decodeStream(it) }
    InstrumentationRegistry.getInstrumentation().runOnMainSync { vm.setPhoto(bmp); vm.update { CHILD_ECZEMA }; vm.analyze() }
    withTimeout(90_000) { vm.state.first { it is AnalysisState.Done || it is AnalysisState.Failed } }
}

/** A1a: first run navigation on the real activity (onboarding → setup → home → scan). */
@RunWith(AndroidJUnit4::class)
class FirstRunFlowTest {
    @get:Rule val compose = createEmptyComposeRule()
    @get:Rule val camera: GrantPermissionRule = GrantPermissionRule.grant(Manifest.permission.CAMERA)

    @Before fun freshInstall() {
        app.container.settings.apply { setOnboarded(false); setSetupSeen(false); setHistory(false); setLang("en") }
    }

    @Test fun onboardingSetupHomeScan() {
        ActivityScenario.launch(MainActivity::class.java).use {
            compose.onNodeWithText(s(R.string.load_get_started)).performClick()
            repeat(2) { compose.onNodeWithText(s(R.string.onb_next)).performClick() }
            compose.onNodeWithText(s(R.string.onb_understand)).performClick()
            compose.waitForIdle()
            // with a verified model already on the phone the app skips setup (ModelManager), otherwise skip it here
            if (compose.onAllNodesWithText(s(R.string.setup_skip)).fetchSemanticsNodes().isNotEmpty())
                compose.onNodeWithText(s(R.string.setup_skip)).performClick()
            compose.onNodeWithText(s(R.string.home_scan)).assertIsDisplayed().performClick()
            compose.onNodeWithText(s(R.string.scan_title)).assertIsDisplayed()
        }
        assertTrue(app.container.settings.onboarded.value)
    }
}

/** A1b, A7, A10 on composables driven by a real SessionViewModel + real CV + real rules + fake LLM. */
@RunWith(AndroidJUnit4::class)
class ResultFlowTest {
    @get:Rule val compose = createComposeRule()
    private lateinit var realLlm: Llm

    @Before fun swapLlm() { realLlm = app.container.llm; app.container.llm = FakeLlm(FAKE_ANALYSIS) }
    @After fun restore() { app.container.llm = realLlm; app.container.settings.setHistory(false) }

    /** A1b: result shows the red-flag banner, the categories, the rule-raised tier and the disclaimer. */
    @Test fun resultRendersSafetyFirst() {
        assumeTrue("CV model not bundled", app.container.cv.available)
        val vm = SessionViewModel(app)
        val end = analyzeWithFake(vm)
        assertTrue("analysis failed: $end", end is AnalysisState.Done)
        val r = vm.result.value!!
        assertEquals(Mode.full, r.mode)
        assertTrue("R5 (child) must fire", "rf_r5" in r.ruleMessages)
        assertTrue("rules lift the LLM's LOW", Tier.parse(r.finalTier)!! >= Tier.MODERATE)
        compose.setContent { SkinNovaTheme(false) { ResultScreen(vm, {}, {}, {}, {}) } }
        compose.onNodeWithText(s(R.string.rf_r5), substring = true).assertExists()
        compose.onNodeWithText(s(R.string.cat_eczema_atopic), substring = true).assertExists()
        compose.onNodeWithText(s(R.string.disclaimer)).performScrollTo().assertIsDisplayed()
    }

    /** A7: history is off by default; once on, a result is stored; "delete everything" removes DB rows and image files. */
    @Test fun historyOptInAndDeleteEverything() = runBlocking {
        assumeTrue(app.container.cv.available)
        val c = app.container
        val vm = SessionViewModel(app)
        deleteEverything(vm, alsoModel = false)
        assertEquals("history must be opt-in", false, c.settings.history.value)
        c.settings.setHistory(true)
        analyzeWithFake(vm)
        withTimeout(10_000) { c.db.dao().analysisCount().first { it == 1 } }
        assertTrue("encrypted photo stored", c.images.count() >= 1)
        deleteEverything(vm, alsoModel = false)
        assertEquals(0, c.db.dao().analysisCount().first())
        assertEquals(0, c.images.count())
    }

    /** A10: every clickable element on the result screen is ≥ 48 dp in both dimensions (touch target), at font scale 1.5. */
    @Test fun resultTouchTargetsAndLargeFont() {
        assumeTrue(app.container.cv.available)
        val vm = SessionViewModel(app)
        analyzeWithFake(vm)
        var density = 0f
        compose.setContent {
            val d = androidx.compose.ui.platform.LocalDensity.current
            androidx.compose.runtime.CompositionLocalProvider(androidx.compose.ui.platform.LocalDensity provides Density(d.density, 1.5f)) {
                density = d.density
                SkinNovaTheme(false) { ResultScreen(vm, {}, {}, {}, {}) }
            }
        }
        compose.onNodeWithText(s(R.string.disclaimer)).performScrollTo().assertIsDisplayed()   // nothing clipped away at 1.5×
        val minPx = 48 * density - 1
        val small = compose.onAllNodes(hasClickAction()).fetchSemanticsNodes().filter { n ->
            n.boundsInRoot.width < minPx || n.boundsInRoot.height < minPx
        }.map { n ->
            val label = n.config.getOrNull(SemanticsProperties.Text)?.joinToString { it.text }
                ?: n.config.getOrNull(SemanticsProperties.ContentDescription)?.joinToString() ?: "?"
            "$label ${(n.boundsInRoot.width / density).toInt()}x${(n.boundsInRoot.height / density).toInt()}dp"
        }
        assertTrue("touch targets below 48 dp: $small", small.isEmpty())
    }
}

