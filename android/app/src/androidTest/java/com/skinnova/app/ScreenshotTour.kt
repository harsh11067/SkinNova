package com.skinnova.app

import android.app.Application
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.compose.ui.graphics.asAndroidBitmap
import androidx.compose.ui.test.captureToImage
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onAllNodesWithText
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.onRoot
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performScrollTo
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.skinnova.app.ui.Draft
import com.skinnova.app.ui.SessionViewModel
import com.skinnova.app.ui.TimelineViewModel
import com.skinnova.app.ui.screens.LibraryScreen
import com.skinnova.app.ui.screens.LockScreen
import com.skinnova.app.ui.screens.ProfileEditScreen
import com.skinnova.app.ui.screens.ProfileScreen
import com.skinnova.app.ui.theme.SkinNovaTheme
import org.junit.Assume.assumeTrue
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

private val app get() = ApplicationProvider.getApplicationContext<Application>() as SkinNovaApp
private fun s(id: Int) = app.getString(id)

/**
 * Visual QA on the phone: real model, real screens, captured as PNGs in files/screens (pulled and reviewed by eye).
 * Not a pass/fail gate beyond "the screen rendered"; run with `-e class com.skinnova.app.ScreenshotTour`.
 */
@RunWith(AndroidJUnit4::class)
class ScreenshotTour {
    @get:Rule val compose = createComposeRule()
    private val dir = File(app.getExternalFilesDir("screens"), "").apply { mkdirs() }
    private fun shot(name: String) {
        compose.waitForIdle()
        val b = compose.onRoot().captureToImage().asAndroidBitmap()
        File(dir, "$name.png").outputStream().use { b.compress(Bitmap.CompressFormat.PNG, 100, it) }
    }
    private fun dark() = app.container.settings.dark.value

    /** Real analysis on the GPU: Early look while writing → result → home care → Ask SkinNova → Learn more. */
    @Test fun analysisTour() {
        assumeTrue(app.container.cv.available && app.container.llm.available)
        kotlinx.coroutines.runBlocking { requireNoPersonalData() }
        val vm = SessionViewModel(app)
        val bmp = InstrumentationRegistry.getInstrumentation().context.assets.open("cv_fixtures/03.jpg").use { BitmapFactory.decodeStream(it) }
        InstrumentationRegistry.getInstrumentation().runOnMainSync {
            vm.setPhoto(bmp); vm.update { Draft("arm", "1_4w", 2, 1, "no", false, false, false, false, "18_39") }
        }
        compose.waitUntil(20_000) { vm.cv.value.isNotEmpty() }
        InstrumentationRegistry.getInstrumentation().runOnMainSync { vm.analyze() }
        compose.setContent { SkinNovaTheme(dark()) { App(vm, TimelineViewModel(app), startRoute = Routes.ANALYZING) } }
        compose.waitUntil(30_000) { compose.onAllNodesWithText(s(R.string.an_early_title)).fetchSemanticsNodes().isNotEmpty() }
        shot("01_analyzing_early_look")
        compose.waitUntil(300_000) { compose.onAllNodesWithText(s(R.string.res_title)).fetchSemanticsNodes().isNotEmpty() }
        shot("02_result_top")
        compose.onNodeWithText(s(R.string.res_explanation)).performScrollTo(); shot("03_result_explanation")
        compose.onNodeWithText(s(R.string.rl_title)).performScrollTo(); shot("04_home_care_relief")
        // the suggestion chip (the home-care card now also has an "Is it contagious?" line, listed first)
        compose.onAllNodesWithText(s(R.string.ask_s_contagious)).let { it[it.fetchSemanticsNodes().size - 1] }.performScrollTo().performClick()
        compose.waitUntil(120_000) { vm.chat.value.lastOrNull()?.answer != null }
        compose.onNodeWithText(s(R.string.ask_title)).performScrollTo(); shot("05_ask_answer")
        compose.onNodeWithText(s(R.string.res_learn)).performScrollTo(); shot("06_result_buttons")
        compose.onNodeWithText(s(R.string.res_learn)).performClick()
        compose.waitUntil(10_000) { compose.onAllNodesWithText(s(R.string.ins_title)).fetchSemanticsNodes().isNotEmpty() }
        shot("07_insights_photos")
    }

    @Test fun libraryProfileLock() {
        val vm = SessionViewModel(app)
        val pageState = androidx.compose.runtime.mutableStateOf(0)
        compose.setContent {
            SkinNovaTheme(dark()) {
                when (pageState.value) {
                    0 -> LibraryScreen(vm) {}
                    1 -> ProfileScreen(vm, {}, {})
                    2 -> ProfileEditScreen(vm) {}
                    else -> LockScreen(app.container.lock) {}
                }
            }
        }
        shot("10_library")
        pageState.value = 1; shot("11_profile")
        compose.onNodeWithText(s(R.string.sec_title)).performScrollTo(); shot("12_profile_security")
        pageState.value = 2; shot("13_profile_edit")
        pageState.value = 3; shot("14_lock_screen")
    }
}
