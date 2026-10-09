package com.skinnova.app.ui.screens

import android.text.format.Formatter
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.requiredSize
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Shadow
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.TextUnit
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.skinnova.app.BuildConfig
import com.skinnova.app.R
import com.skinnova.app.di.AppContainer
import com.skinnova.app.setup.ImportState
import com.skinnova.app.setup.ModelDownload
import com.skinnova.app.ui.components.Chip
import com.skinnova.app.ui.components.OutlineButton
import com.skinnova.app.ui.components.PixelImage
import com.skinnova.app.ui.components.PrimaryButton
import com.skinnova.app.ui.components.SnCard
import com.skinnova.app.ui.theme.LocalSn
import com.skinnova.app.ui.theme.Pixelify
import com.skinnova.app.ui.theme.PlexMono
import com.skinnova.app.ui.theme.SnType
import kotlinx.coroutines.launch

/**
 * 00 Loading — design index.html "00 Loading" (a 360×788 CSS-px frame). The scene is the design's own procedural canvas
 * drawn as a 180×394 dot grid (px_landing_grid + PixelDotArt: whole pixels per dot, 15 fps star twinkle) and every element sits at its design
 * coordinate × k, k = the art's cover scale — so text, logo and controls land on the art exactly as designed.
 * Real work underneath: verify any adb-pushed model once (sha256 is cached afterwards), then show Get Started.
 */
@Composable
fun LoadingScreen(c: AppContainer, onDone: () -> Unit) {
    var progress by remember { mutableFloatStateOf(0f) }
    var verified by remember { mutableStateOf<Boolean?>(null) }
    var ready by remember { mutableStateOf(false) }
    LaunchedEffect(Unit) {
        val m = c.models.scanAndVerify { progress = it * 0.95f }
        verified = m != null
        progress = 1f; ready = true
    }
    val words = listOf(R.string.load_w_observe, R.string.load_w_analyze, R.string.load_w_understand, R.string.load_w_act)
    BoxWithConstraints(Modifier.fillMaxSize().background(Color(0xFF060A17))) {
        val density = LocalDensity.current
        // art = 180×394 dot grid at the largest WHOLE number of screen pixels per dot that fits (crisp on every phone;
        // 1080×2400 → 6 px dots = the full 1080 px width; rounding up instead zoomed 15 % and cut the logo off)
        val cell = minOf(constraints.maxWidth / 180, constraints.maxHeight / 394).coerceAtLeast(1)
        val k = cell * 180f / density.density / 360f                // design px → dp (the 360×788 design box = the art)
        fun d(px: Float): Dp = (px * k).dp
        val ctx = androidx.compose.ui.platform.LocalContext.current
        val art by androidx.compose.runtime.produceState<com.skinnova.app.ui.components.DotArt?>(null, cell) {
            value = com.skinnova.app.ui.components.buildDotArt(ctx, R.drawable.px_landing_grid, cell)
        }
        fun t(px: Float): TextUnit = with(density) { d(px).toSp() }    // text scales with the art, like the design
        Box(Modifier.align(Alignment.Center).requiredSize(d(360f), d(788f))) {
            art?.let { com.skinnova.app.ui.components.DotArtCanvas(it, Modifier.fillMaxSize()) }
                ?: PixelImage(R.drawable.px_landing_grid, Modifier.fillMaxSize(), ContentScale.FillBounds)   // for the ~0.1 s it takes to build
            Box(Modifier.align(Alignment.BottomStart).fillMaxWidth().height(d(260f))
                .background(Brush.verticalGradient(0f to Color(0x00060A17), 0.45f to Color(0xB3060A17), 1f to Color(0xF5060A17))))
            // shared logo on this screen: translate(14px,100px) scale(1.06) — gem 48 · gap 10 · title 231×47
            Row(Modifier.offset(d(14f), d(100f)), verticalAlignment = Alignment.CenterVertically) {
                PixelImage(R.drawable.px_gem, Modifier.size(d(48f * 1.06f)), ContentScale.Fit)
                Spacer(Modifier.width(d(10f * 1.06f)))
                PixelImage(R.drawable.landing_title, Modifier.size(d(231f * 1.06f), d(47f * 1.06f)), ContentScale.Fit)
            }
            Text(stringResource(R.string.load_tagline), Modifier.offset(d(30f), d(180f)), color = Color(0xFFF3ECE4), fontFamily = PlexMono,
                fontSize = t(14f), style = TextStyle(shadow = Shadow(Color.Black, Offset(0f, 2f), 6f)))
            Text(stringResource(R.string.load_sub), Modifier.offset(d(30f), d(206f)), color = Color(0xFFD8D2EA), fontFamily = PlexMono,
                fontSize = t(12f), lineHeight = t(19f))
            Column(Modifier.align(Alignment.BottomStart).padding(start = d(28f), bottom = d(150f)), verticalArrangement = Arrangement.spacedBy(d(4f))) {
                words.forEachIndexed { i, w ->
                    val lit = progress >= (i + 1) / 4f - 0.01f
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Box(Modifier.size(d(6f)).clip(RoundedCornerShape(d(2f))).background(if (lit) Color(0xFFF2C9A1) else Color(0xFF3A3570)))
                        Spacer(Modifier.width(d(10f)))
                        Text(stringResource(w), color = if (lit) Color(0xFFF3ECE4) else Color(0xFF6D69A0), fontFamily = PlexMono,
                            fontSize = t(11f), letterSpacing = t(2.4f))
                    }
                }
            }
            if (!ready) {
                Column(Modifier.align(Alignment.BottomStart).fillMaxWidth().padding(start = d(28f), end = d(28f), bottom = d(58f))) {
                    Row(Modifier.fillMaxWidth().padding(bottom = d(10f))) {
                        Text(stringResource(R.string.load_checking), color = Color(0xFFD8D2EA), fontFamily = PlexMono, fontSize = t(10.5f), modifier = Modifier.weight(1f))
                        Text("${(progress * 100).toInt()}%", color = Color(0xFFF2C9A1), fontFamily = PlexMono, fontSize = t(10.5f))
                    }
                    Row(Modifier.fillMaxWidth().clip(RoundedCornerShape(d(10f))).background(Color(0x99060A17))
                        .border(1.dp, Color(0xFF57539A), RoundedCornerShape(d(10f))).padding(d(5f)), horizontalArrangement = Arrangement.spacedBy(d(3f))) {
                        repeat(24) { i ->
                            Box(Modifier.weight(1f).height(d(8f)).clip(RoundedCornerShape(d(2f))).background(if (progress * 24 > i) Color(0xFFEAB88C) else Color(0xFF1C1A3A)))
                        }
                    }
                    Text(stringResource(R.string.load_footer), color = Color(0xFFA9A5C6), fontFamily = PlexMono, fontSize = t(9.5f), letterSpacing = t(1.6f),
                        textAlign = TextAlign.Center, modifier = Modifier.fillMaxWidth().padding(top = d(12f)))
                }
            } else {
                Row(Modifier.align(Alignment.BottomStart).padding(start = d(28f), end = d(28f), bottom = d(62f)).fillMaxWidth().height(d(54f))
                    .clip(RoundedCornerShape(d(18f))).background(Color(0xD10A0D1B)).border(1.5.dp, Color(0xFFEAB88C), RoundedCornerShape(d(18f)))
                    .clickable(role = Role.Button, onClick = onDone).semantics { contentDescription = "Get Started" },
                    horizontalArrangement = Arrangement.Center, verticalAlignment = Alignment.CenterVertically) {
                    Text(stringResource(R.string.load_get_started), color = Color(0xFFF4DCC2), fontFamily = Pixelify, fontSize = t(17f), letterSpacing = t(0.6f))
                    Spacer(Modifier.width(d(12f)))
                    Text("→", color = Color(0xFFF2C9A1), fontSize = t(19f))
                }
                if (verified == false) Text(stringResource(R.string.load_no_model), color = Color(0xFFD8D2EA), fontFamily = PlexMono, fontSize = t(10.5f),
                    textAlign = TextAlign.Center, modifier = Modifier.align(Alignment.BottomCenter).padding(bottom = d(30f)))
            }
        }
    }
}

/** Onboarding: 3 cards (what it does · private & offline · not a diagnosis) + language + "I understand". */
@Composable
fun OnboardingScreen(c: AppContainer, onDone: () -> Unit) {
    val sn = LocalSn.current
    var page by remember { mutableIntStateOf(0) }
    val lang by c.settings.lang.collectAsState()
    val pages = listOf(R.string.onb_1_title to R.string.onb_1_body, R.string.onb_2_title to R.string.onb_2_body, R.string.onb_3_title to R.string.onb_3_body)
    val arts = listOf(R.drawable.px_scan, R.drawable.px_splash, R.drawable.px_insights)
    Column(Modifier.fillMaxSize().background(sn.bgBrush).statusBarsPadding().navigationBarsPadding().padding(18.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text(stringResource(R.string.onb_language), style = SnType.caption, color = sn.mut)
            Spacer(Modifier.width(10.dp))
            Chip("English", lang == "en", { c.settings.setLang("en") })
            Spacer(Modifier.width(8.dp))
            Chip("हिन्दी", lang == "hi", { c.settings.setLang("hi") })
        }
        Spacer(Modifier.height(18.dp))
        Box(Modifier.fillMaxWidth().height(260.dp).clip(RoundedCornerShape(26.dp)).border(1.5.dp, sn.acc, RoundedCornerShape(26.dp)).padding(5.dp)) {
            PixelImage(arts[page], Modifier.fillMaxSize().clip(RoundedCornerShape(21.dp)))
        }
        Spacer(Modifier.height(22.dp))
        Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            repeat(3) { i -> Box(Modifier.weight(1f).height(6.dp).clip(RoundedCornerShape(3.dp)).background(if (i <= page) sn.acc else sn.track)) }
        }
        Spacer(Modifier.height(22.dp))
        Text(stringResource(pages[page].first), style = SnType.display, color = sn.ink)
        Spacer(Modifier.height(10.dp))
        Text(stringResource(pages[page].second), style = SnType.bodyL, color = sn.mut)
        Spacer(Modifier.weight(1f))
        if (page < 2) PrimaryButton(stringResource(R.string.onb_next)) { if (page < 2) page++ }   // re-check: double tap in one frame
        else PrimaryButton(stringResource(R.string.onb_understand)) { c.settings.setOnboarded(true); onDone() }
    }
}

/** Model setup (one-time): import via SAF (primary), Wi-Fi download (online flavor only), or continue in Basic mode. */
@Composable
fun SetupScreen(c: AppContainer, onDone: () -> Unit) {
    val sn = LocalSn.current
    val ctx = LocalContext.current
    val scope = rememberCoroutineScope()
    var st by remember { mutableStateOf<ImportState>(if (c.models.activeModel() != null) ImportState.Ready(c.models.activeModel()!!) else ImportState.Idle) }
    val picker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) scope.launch { c.models.import(uri) { st = it } }
    }
    // A model copied into Android/data/…/files/models (adb, USB file transfer) is not reachable from the system picker:
    // hash it here (the splash only scans at app start), on entry and on demand. ≥ 0 while hashing.
    var scan by remember { mutableFloatStateOf(-1f) }
    var folderMismatch by remember { mutableStateOf(false) }
    suspend fun scanFolder() {
        if (st is ImportState.Ready || scan >= 0f) return
        scan = 0f
        val m = c.models.scanAndVerify { scan = it }
        scan = -1f
        if (m != null) st = ImportState.Ready(m) else folderMismatch = c.models.hasModelFile()
    }
    LaunchedEffect(Unit) { scanFolder() }
    val need = Formatter.formatShortFileSize(ctx, c.models.requiredBytes)
    Column(Modifier.fillMaxSize().background(sn.bgBrush).statusBarsPadding().navigationBarsPadding().padding(18.dp).verticalScroll(rememberScrollState())) {
        Spacer(Modifier.height(24.dp))
        Text(stringResource(R.string.setup_title), style = SnType.headline, color = sn.ink)
        Spacer(Modifier.height(10.dp))
        Text(stringResource(R.string.setup_body, Formatter.formatShortFileSize(ctx, c.models.modelBytes)), style = SnType.bodyL, color = sn.mut)
        Spacer(Modifier.height(18.dp))
        SnCard {
            Column {
                Text(stringResource(R.string.setup_storage, need, Formatter.formatShortFileSize(ctx, c.models.freeBytes())), style = SnType.body, color = sn.mut)
                Spacer(Modifier.height(14.dp))
                if (scan >= 0f) {
                    Text(stringResource(R.string.setup_verifying), style = SnType.label, color = sn.ink)
                    Spacer(Modifier.height(8.dp)); com.skinnova.app.ui.components.Bar(scan, sn.acc)
                } else if (folderMismatch && st is ImportState.Idle) {
                    Text(stringResource(R.string.setup_err_folder), style = SnType.label, color = sn.urgent)
                }
                when (val s = st) {
                    is ImportState.Idle -> {}
                    is ImportState.Copying -> {
                        Text(stringResource(R.string.setup_copying, (s.fraction * 100).toInt()), style = SnType.label, color = sn.ink)
                        Spacer(Modifier.height(8.dp)); com.skinnova.app.ui.components.Bar(s.fraction, sn.acc)
                    }
                    is ImportState.Verifying -> Text(stringResource(R.string.setup_verifying), style = SnType.label, color = sn.ink)
                    is ImportState.Ready -> Text("✓ " + stringResource(R.string.setup_ready) + " · " + s.model.label, style = SnType.label, color = sn.low)
                    is ImportState.Error -> Text(when (s.kind) {
                        ImportState.Error.Kind.SPACE -> stringResource(R.string.setup_err_space, need)
                        ImportState.Error.Kind.WRONG_FILE -> stringResource(R.string.setup_err_wrong)
                        ImportState.Error.Kind.IO -> stringResource(R.string.setup_err_io, s.detail)
                    }, style = SnType.label, color = sn.urgent)
                }
            }
        }
        Spacer(Modifier.height(16.dp))
        if (st is ImportState.Ready) PrimaryButton(stringResource(R.string.q_next)) { c.settings.setSetupSeen(true); onDone() }
        else {
            PrimaryButton(stringResource(R.string.setup_import), enabled = st !is ImportState.Copying && st !is ImportState.Verifying && scan < 0f) {
                c.lock.expectExternal(); picker.launch(arrayOf("application/octet-stream", "*/*"))
            }
            Spacer(Modifier.height(10.dp))
            OutlineButton(stringResource(R.string.setup_rescan), Modifier.fillMaxWidth()) { scope.launch { scanFolder() } }
            if (ModelDownload.available) {
                Spacer(Modifier.height(10.dp))
                OutlineButton(stringResource(R.string.setup_download), Modifier.fillMaxWidth()) {
                    scope.launch { st = ModelDownload.download(ctx, c.models) { st = it } }
                }
            }
            Spacer(Modifier.height(10.dp))
            OutlineButton(stringResource(R.string.setup_skip), Modifier.fillMaxWidth()) { c.settings.setSetupSeen(true); onDone() }
            Spacer(Modifier.height(10.dp))
            Text(stringResource(R.string.setup_skip_hint), style = SnType.caption, color = sn.mut)
        }
        if (BuildConfig.DEBUG) {
            Spacer(Modifier.height(18.dp))
            Text(stringResource(R.string.setup_adb_hint), style = SnType.micro, color = sn.mut)
        }
    }
}

/**
 * One-time "Optimising SkinNova for your phone" (v2.1 item 5): after a model import or an app update, LiteRT-LM builds
 * the GPU weight cache (~2 min on a mid-range phone). Done here, in the open, instead of inside the first analysis.
 */
@Composable
fun OptimizeScreen(vm: com.skinnova.app.ui.SessionViewModel, onDone: () -> Unit) {
    val sn = LocalSn.current
    val c = vm.c
    var elapsed by remember { mutableFloatStateOf(0f) }
    var done by remember { mutableStateOf(c.engineHolder.isLoaded) }
    LaunchedEffect(Unit) {
        vm.warmUp()   // viewModelScope: keeps going if the user leaves this screen
        val t0 = System.currentTimeMillis()
        while (!c.engineHolder.isLoaded) { kotlinx.coroutines.delay(500); elapsed = (System.currentTimeMillis() - t0) / 1000f
            if (elapsed > 600f) break }
        done = c.engineHolder.isLoaded
        if (done) { c.optimizeKey()?.let { c.settings.setOptimizedKey(it) }; kotlinx.coroutines.delay(700); onDone() }
    }
    Column(Modifier.fillMaxSize().background(sn.bgBrush).statusBarsPadding().navigationBarsPadding().padding(24.dp),
        horizontalAlignment = Alignment.CenterHorizontally) {
        Spacer(Modifier.height(60.dp))
        PixelImage(R.drawable.px_gem, Modifier.size(64.dp), ContentScale.Fit)
        Spacer(Modifier.height(20.dp))
        Text(stringResource(R.string.opt_title), style = SnType.headline, color = sn.ink, textAlign = TextAlign.Center)
        Spacer(Modifier.height(10.dp))
        Text(stringResource(R.string.opt_sub), style = SnType.body, color = sn.mut, textAlign = TextAlign.Center)
        Spacer(Modifier.height(28.dp))
        com.skinnova.app.ui.components.PixelProgressBar(if (done) 1f else (elapsed / 120f).coerceIn(0.02f, 0.95f))
        Spacer(Modifier.height(10.dp))
        Text(if (done) stringResource(R.string.opt_done) else stringResource(R.string.opt_elapsed, elapsed.toInt()), style = SnType.caption, color = sn.accT)
        Spacer(Modifier.height(24.dp))
        com.skinnova.app.ui.screens.KeepRunningTip()
        Spacer(Modifier.weight(1f))
        TextButton(onClick = onDone) { Text(stringResource(R.string.opt_later), style = SnType.caption, color = sn.mut) }
    }
}
