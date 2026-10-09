package com.skinnova.app

import android.content.Context
import android.content.res.Configuration
import android.os.Bundle
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.viewModels
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.clearAndSetSemantics
import kotlinx.coroutines.launch
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.navigation.NavHostController
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.rememberNavController
import com.skinnova.app.ui.SessionViewModel
import com.skinnova.app.ui.TimelineViewModel
import com.skinnova.app.ui.screens.AnalyzingScreen
import com.skinnova.app.ui.screens.HistoryScreen
import com.skinnova.app.ui.screens.HomeScreen
import com.skinnova.app.ui.screens.InsightsScreen
import com.skinnova.app.ui.screens.LibraryScreen
import com.skinnova.app.ui.screens.LoadingScreen
import com.skinnova.app.ui.screens.OnboardingScreen
import com.skinnova.app.ui.screens.ProfileScreen
import com.skinnova.app.ui.screens.ProfileEditScreen
import com.skinnova.app.ui.screens.SetPinScreen
import com.skinnova.app.ui.screens.LockScreen
import com.skinnova.app.ui.screens.QuestionsScreen
import com.skinnova.app.ui.screens.RecaptureScreen
import com.skinnova.app.ui.screens.ResultScreen
import com.skinnova.app.ui.screens.ScanScreen
import com.skinnova.app.ui.screens.SetupScreen
import com.skinnova.app.ui.screens.TimelineScreen
import com.skinnova.app.ui.screens.TrackSpotScreen
import com.skinnova.app.ui.theme.LocalSn
import com.skinnova.app.ui.theme.SkinNovaTheme
import com.skinnova.app.ui.theme.SnType
import java.util.Locale

/** FragmentActivity (not plain ComponentActivity): BiometricPrompt needs it for fingerprint unlock. */
class MainActivity : androidx.fragment.app.FragmentActivity() {
    private val vm: SessionViewModel by viewModels()
    private val tvm: TimelineViewModel by viewModels()

    /** Per-app language from Settings (works on every API level; Hindi strings in values-hi). */
    override fun attachBaseContext(base: Context) {
        val lang = base.getSharedPreferences("settings", MODE_PRIVATE).getString("lang", "en") ?: "en"
        val cfg = Configuration(base.resources.configuration).apply { setLocale(Locale(lang)) }
        super.attachBaseContext(base.createConfigurationContext(cfg))
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        val c = (application as SkinNovaApp).container
        setContent {
            val dark by c.settings.dark.collectAsState()
            val lockOn by c.lock.on.collectAsState()
            // with an app lock, keep health data out of screenshots and the recent-apps thumbnail
            LaunchedEffect(lockOn) {
                if (lockOn) window.setFlags(android.view.WindowManager.LayoutParams.FLAG_SECURE, android.view.WindowManager.LayoutParams.FLAG_SECURE)
                else window.clearFlags(android.view.WindowManager.LayoutParams.FLAG_SECURE)
            }
            val lang by c.settings.lang.collectAsState()
            val initialLang = androidx.compose.runtime.remember { lang }
            LaunchedEffect(lang) { if (lang != initialLang) recreate() }
            SkinNovaTheme(dark) {
                val sn = LocalSn.current
                App(vm, tvm)
            }
        }
    }
}

object Routes {
    const val LOADING = "loading"; const val ONBOARD = "onboarding"; const val SETUP = "setup"; const val HOME = "home"
    const val SCAN = "scan"; const val QUESTIONS = "questions"; const val ANALYZING = "analyzing"; const val RESULT = "result"
    const val INSIGHTS = "insights/{key}"; const val LIBRARY = "library"; const val HISTORY = "history"; const val PROFILE = "profile"
    const val TRACK = "track"; const val TIMELINE = "timeline/{spotId}"; const val RECAPTURE = "recapture/{spotId}"
    const val PROFILE_EDIT = "profile/edit"; const val SET_PIN = "security/pin/{mode}"; const val OPTIMIZE = "optimize"
    val TABS = setOf(HOME, HISTORY, LIBRARY, PROFILE)
}

@Composable
fun App(vm: SessionViewModel, tvm: TimelineViewModel, startRoute: String = Routes.LOADING) {   // startRoute: on-device nav tests
    val nav = rememberNavController()
    val c = vm.c
    val entry by nav.currentBackStackEntryAsState()
    val route = entry?.destination?.route
    fun homeOrOptimize() = if (c.needsOptimizing()) Routes.OPTIMIZE else Routes.HOME
    fun afterStart() {
        val next = when {
            !c.settings.onboarded.value -> Routes.ONBOARD
            c.models.activeModelPath() == null && !c.settings.setupSeen.value -> Routes.SETUP
            else -> homeOrOptimize()
        }
        // plenty of free memory: start loading the model now (v2.1 item 5); otherwise at photo review
        if (next == Routes.HOME && availMemGb(c.ctx) >= 3.5) vm.warmUp()
        nav.navigate(next) { popUpTo(Routes.LOADING) { inclusive = true } }
    }
    val locked by c.lock.locked.collectAsState()
    val scope = androidx.compose.runtime.rememberCoroutineScope()
    // on lock: drop keyboard focus (typing must not reach a hidden field) and stop any read-aloud of health text
    val focus = androidx.compose.ui.platform.LocalFocusManager.current
    val keyboard = androidx.compose.ui.platform.LocalSoftwareKeyboardController.current
    LaunchedEffect(locked) { if (locked) { focus.clearFocus(force = true); keyboard?.hide(); com.skinnova.app.i18n.Tts.stop() } }
    Box(Modifier.fillMaxSize().background(LocalSn.current.bgBrush)) {
      // Route guard: while locked no route is reachable — the lock screen covers the whole NavHost, takes every touch,
      // and the routes underneath are removed from the accessibility tree (FLAG_SECURE hides them from screenshots).
      // LocalAppLocked: screens' dialogs (own windows, above the lock screen) are not shown while locked
      androidx.compose.runtime.CompositionLocalProvider(com.skinnova.app.ui.LocalAppLocked provides locked) {
      Box(Modifier.fillMaxSize().then(if (locked) Modifier.clearAndSetSemantics { } else Modifier)) {
        NavHost(nav, startDestination = startRoute) {
            composable(Routes.LOADING) { LoadingScreen(c) { afterStart() } }
            composable(Routes.ONBOARD) { OnboardingScreen(c) {
                nav.navigate(if (c.models.activeModelPath() == null && !c.settings.setupSeen.value) Routes.SETUP else homeOrOptimize()) { popUpTo(0) }
            } }
            composable(Routes.SETUP) { SetupScreen(c) { nav.navigate(homeOrOptimize()) { popUpTo(0) } } }
            composable(Routes.OPTIMIZE) { com.skinnova.app.ui.screens.OptimizeScreen(vm) { nav.navigate(Routes.HOME) { popUpTo(0) } } }
            composable(Routes.HOME) {
                HomeScreen(vm, onScan = { nav.navigate(Routes.SCAN) }, onHistory = { nav.tab(Routes.HISTORY) }, onLibrary = { nav.tab(Routes.LIBRARY) },
                    onSpot = { nav.navigate("timeline/$it") }, onOpenRecent = { vm.openSaved(it); nav.navigate(Routes.RESULT) })
            }
            composable(Routes.SCAN) { ScanScreen(vm, onBack = { nav.popBackStack() }, onPhotoAccepted = { nav.navigate(Routes.QUESTIONS) }) }
            composable(Routes.QUESTIONS) { QuestionsScreen(vm, onBack = { nav.popBackStack() }, onAnalyze = { vm.analyze(); nav.navigate(Routes.ANALYZING) }) }
            composable(Routes.ANALYZING) {
                AnalyzingScreen(vm, onDone = { nav.navigate(Routes.RESULT) { popUpTo(Routes.HOME) } }, onCancel = { nav.popBackStack() })
            }
            composable(Routes.RESULT) {
                ResultScreen(vm, onHome = { nav.navigate(Routes.HOME) { popUpTo(0) } }, onRetake = { nav.navigate(Routes.SCAN) { popUpTo(Routes.HOME) } },
                    onLearn = { nav.navigate("insights/$it") }, onTrack = { nav.navigate(Routes.TRACK) })
            }
            composable(Routes.INSIGHTS) { e -> InsightsScreen(vm, e.arguments?.getString("key") ?: "other") { nav.popBackStack() } }
            composable(Routes.LIBRARY) { LibraryScreen(vm) { nav.navigate("insights/$it") } }
            composable(Routes.HISTORY) {
                HistoryScreen(vm, onOpen = { vm.openSaved(it); nav.navigate(Routes.RESULT) }, onSpot = { nav.navigate("timeline/$it") }, onScan = { nav.navigate(Routes.SCAN) })
            }
            composable(Routes.PROFILE) { ProfileScreen(vm, onReplayIntro = { c.settings.setOnboarded(false); nav.navigate(Routes.ONBOARD) { popUpTo(0) } },
                onSetup = { nav.navigate(Routes.SETUP) }, onEdit = { nav.navigate(Routes.PROFILE_EDIT) }, onPin = { mode -> nav.navigate("security/pin/$mode") },
                onStart = { nav.navigate(Routes.LOADING) { popUpTo(nav.graph.id) { inclusive = true } } }) }
            composable(Routes.PROFILE_EDIT) { ProfileEditScreen(vm) { nav.popBackStack() } }
            composable(Routes.SET_PIN) { e -> SetPinScreen(c.lock, e.arguments?.getString("mode") ?: "new", onDone = { nav.popBackStack() }, onBack = { nav.popBackStack() }) }
            composable(Routes.TRACK) { TrackSpotScreen(vm, tvm, onBack = { nav.popBackStack() }, onCreated = { nav.navigate("timeline/$it") { popUpTo(Routes.HOME) } }) }
            composable(Routes.TIMELINE) { e -> TimelineScreen(vm, tvm, e.arguments?.getString("spotId")!!, onBack = { nav.popBackStack() },
                onRecapture = { nav.navigate("recapture/$it") }) }
            composable(Routes.RECAPTURE) { e -> RecaptureScreen(vm, tvm, e.arguments?.getString("spotId")!!, onDone = { nav.popBackStack() }) }
        }
        if (route in Routes.TABS) BottomNav(route!!) { nav.tab(it) }
      }
      }
        if (locked) LockScreen(c.lock, onForgot = {
            // no PIN recovery exists offline: the way back in is a reset of the personal data (the AI model stays)
            scope.launch {
                com.skinnova.app.ui.screens.deleteEverything(vm, alsoModel = false)
                vm.clearSession()                        // nothing of the last analysis stays in memory either
                com.skinnova.app.security.BiometricKey.delete()
                c.lock.disable()
                nav.navigate(Routes.HOME) { popUpTo(0) }
            }
        })
    }
}

private fun NavHostController.tab(r: String) = navigate(r) { popUpTo(Routes.HOME) { saveState = true }; launchSingleTop = true; restoreState = true }

/** Design bottom nav: 4 tabs, active colour + underline bar. */
@Composable
fun BottomNav(current: String, go: (String) -> Unit) {
    val sn = LocalSn.current
    Box(Modifier.fillMaxSize(), contentAlignment = Alignment.BottomCenter) {
        Row(Modifier.navigationBarsPadding().padding(start = 12.dp, end = 12.dp, bottom = 12.dp).fillMaxWidth().height(66.dp)
            .then(if (sn.isDark) Modifier else Modifier.shadow(4.dp, RoundedCornerShape(24.dp))).clip(RoundedCornerShape(24.dp)).background(sn.nav).border(1.dp, sn.line, RoundedCornerShape(24.dp))) {
            listOf(Triple(Routes.HOME, R.string.nav_home, "⌂"), Triple(Routes.HISTORY, R.string.nav_history, "▤"),
                Triple(Routes.LIBRARY, R.string.nav_library, "✿"), Triple(Routes.PROFILE, R.string.nav_profile, "◯")).forEach { (r, l, g) ->
                val on = current == r
                Column(Modifier.weight(1f).fillMaxSize().clickable(role = Role.Tab) { go(r) }, horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = androidx.compose.foundation.layout.Arrangement.Center) {
                    Text(g, color = if (on) sn.accT else sn.mut, fontSize = 17.sp)
                    Text(stringResource(l), style = SnType.micro, color = if (on) sn.accT else sn.mut)
                    Spacer(Modifier.height(4.dp))
                    Box(Modifier.width(18.dp).height(3.dp).clip(RoundedCornerShape(2.dp)).background(if (on) sn.accT else sn.nav))
                }
            }
        }
    }
}

/** Free RAM in GB (ActivityManager.MemoryInfo.availMem). */
fun availMemGb(ctx: Context): Double {
    val mi = android.app.ActivityManager.MemoryInfo()
    ctx.getSystemService(android.app.ActivityManager::class.java).getMemoryInfo(mi)
    return mi.availMem / 1e9
}
