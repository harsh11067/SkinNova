package com.skinnova.app

import android.Manifest
import android.app.Application
import android.app.NotificationManager
import android.graphics.BitmapFactory
import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.assertIsEnabled
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onAllNodesWithText
import androidx.compose.ui.test.onNodeWithContentDescription
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performScrollTo
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.rule.GrantPermissionRule
import com.skinnova.app.ml.AnalysisState
import com.skinnova.app.ml.Llm
import com.skinnova.app.model.Tier
import com.skinnova.app.notify.Notifier
import com.skinnova.app.ui.Draft
import com.skinnova.app.ui.SessionViewModel
import com.skinnova.app.ui.TimelineViewModel
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

private val app get() = ApplicationProvider.getApplicationContext<Application>() as SkinNovaApp
private fun s(id: Int) = app.getString(id)

/**
 * The real NavHost (App) from the result screen: every action button must reach its screen (user report 2026-10-08:
 * "Turn on history to save / Track this spot / Retake / Learn more don't work"), and the app lock must hide every route.
 */
@RunWith(AndroidJUnit4::class)
class NavFlowTest {
    @get:Rule val compose = createComposeRule()
    @get:Rule val perms: GrantPermissionRule = GrantPermissionRule.grant(Manifest.permission.CAMERA, Manifest.permission.POST_NOTIFICATIONS)
    private lateinit var realLlm: Llm
    private lateinit var vm: SessionViewModel

    @Before fun setUp() = runBlocking {
        assumeTrue("CV model not bundled", app.container.cv.available)
        realLlm = app.container.llm
        app.container.llm = FakeLlm(FAKE_ANALYSIS_NAV)
        vm = SessionViewModel(app)
        deleteEverything(vm, alsoModel = false)
        app.container.settings.apply { setHistory(false); setLang("en") }
        app.container.lock.disable()
        val bmp = InstrumentationRegistry.getInstrumentation().context.assets.open("cv_fixtures/00.jpg").use { BitmapFactory.decodeStream(it) }
        InstrumentationRegistry.getInstrumentation().runOnMainSync {
            vm.setPhoto(bmp); vm.update { Draft("arm", "1_4w", 2, 0, "no", false, false, false, false, "18_39") }; vm.analyze()
        }
        val end = withTimeout(90_000) { vm.state.first { it is AnalysisState.Done || it is AnalysisState.Failed } }
        assertTrue("analysis: $end", end is AnalysisState.Done)
    }

    @After fun tearDown() = runBlocking {
        app.container.llm = realLlm; app.container.lock.disable(); app.container.settings.setHistory(false)
        deleteEverything(vm, alsoModel = false)
    }

    private fun showResult() = compose.setContent { SkinNovaTheme(false) { App(vm, TimelineViewModel(app), startRoute = Routes.RESULT) } }
    private fun tap(text: Int) = compose.onNodeWithText(s(text)).performScrollTo().performClick()

    @Test fun learnMoreOpensInsights() {
        showResult(); tap(R.string.res_learn)
        compose.onNodeWithText(s(R.string.ins_title)).assertIsDisplayed()
        compose.onNodeWithText(s(R.string.lib_photo_credit)).assertExists()          // real photos on the condition page
    }

    @Test fun saveTurnsOnHistoryAndSaves() = runBlocking {
        showResult(); tap(R.string.res_save)
        compose.onNodeWithText(s(R.string.res_save_title)).assertIsDisplayed()
        compose.onNodeWithText(s(R.string.res_save_ok)).performClick()
        compose.onNodeWithText(s(R.string.res_saved)).assertExists()
        assertTrue(app.container.settings.history.value)
        withTimeout(10_000) { app.container.db.dao().analysisCount().first { it == 1 } }
        Unit
    }

    @Test fun trackThisSpotWorksInOneTap() = runBlocking {
        showResult(); tap(R.string.res_track)
        compose.onNodeWithText(s(R.string.tl_track_title)).assertIsDisplayed()
        compose.onNodeWithText(s(R.string.tl_start)).performScrollTo().assertIsEnabled().performClick()   // pre-filled name
        withTimeout(10_000) { app.container.db.dao().spots().first { it.size == 1 } }
        Unit
    }

    @Test fun retakeOpensCamera() {
        showResult(); tap(R.string.res_retake)
        compose.onNodeWithText(s(R.string.scan_title)).assertIsDisplayed()
    }

    @Test fun reliefAndAskShown() {
        showResult()
        compose.onNodeWithText(s(R.string.rl_title)).performScrollTo().assertIsDisplayed()
        compose.onNodeWithText(s(R.string.ask_title)).performScrollTo().assertIsDisplayed()
    }

    @Test fun lockHidesEveryRouteUntilPin() {
        app.container.lock.setPin("4826"); app.container.lock.lockNow()
        showResult()
        compose.onNodeWithText(s(R.string.lock_title)).assertIsDisplayed()
        assertTrue("result must not be reachable while locked", compose.onAllNodesWithText(s(R.string.res_title)).fetchSemanticsNodes().isEmpty())
        "4826".forEach { compose.onNodeWithText("$it").performClick() }
        compose.waitForIdle()
        assertTrue(compose.onAllNodesWithText(s(R.string.lock_title)).fetchSemanticsNodes().isEmpty())
        compose.onNodeWithText(s(R.string.res_title)).assertIsDisplayed()
    }

    @Test fun resultReadyNotification() {
        Notifier.ready(app, Tier.MODERATE, basic = false)
        val nm = app.getSystemService(NotificationManager::class.java)
        assertTrue(nm.activeNotifications.any { it.id == 42 })
        Notifier.cancelReady(app)
    }
}

/** A validator-valid analysis with a clear eczema lead (LOW): enough for every result-screen section to render. */
private const val FAKE_ANALYSIS_NAV = """{"possible_categories":[{"key":"eczema_atopic","likelihood":"higher","why":"Supported by moderate itching on the arm."},{"key":"contact_dermatitis","likelihood":"possible","why":"Possible: contact with a new product can look similar."},{"key":"psoriasis","likelihood":"less_likely","why":"Less likely: psoriasis plaques are thicker with silvery scale."}],"uncertainty":{"level":"low","reasons":[]},"explanation":"Based on your answers, it has been present for one to four weeks on the arm, with moderate itching. Eczema / atopic dermatitis fits best so far.","what_would_help":["A doctor's examination in person"],"self_care_info":["Moisturise often with a plain, fragrance-free cream."],"triage":{"tier":"LOW","advice":"Watch it and use gentle care."},"disagreement_with_image_model":false}"""
