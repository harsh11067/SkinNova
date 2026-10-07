package com.skinnova.app.ui.screens

import android.text.format.Formatter
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Text
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
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.Role
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
import com.skinnova.app.ui.theme.SnType
import kotlinx.coroutines.launch

/** 00 Loading (design): pixel scene, tagline, word list, 24-segment bar while the app checks the model; then Get Started. */
@Composable
fun LoadingScreen(c: AppContainer, onDone: () -> Unit) {
    var progress by remember { mutableFloatStateOf(0f) }
    var msg by remember { mutableStateOf("Checking on-device model…") }
    var ready by remember { mutableStateOf(false) }
    LaunchedEffect(Unit) {
        // Real work: verify any adb-pushed model once (sha256 is cached afterwards), then show the button.
        val m = c.models.scanAndVerify { progress = it * 0.95f }
        msg = if (m != null) "Model verified" else "Ready"
        progress = 1f; ready = true
    }
    val words = listOf(R.string.load_w_scan, R.string.load_w_learn, R.string.load_w_track, R.string.load_w_care)
    Box(Modifier.fillMaxSize().background(Color(0xFF060A17))) {
        PixelImage(R.drawable.px_loading, Modifier.fillMaxSize(), ContentScale.Crop)
        Box(Modifier.fillMaxSize().background(Brush.verticalGradient(0.55f to Color.Transparent, 1f to Color(0xF5060A17))))
        Column(Modifier.statusBarsPadding().padding(start = 30.dp, top = 60.dp)) {
            PixelImage(R.drawable.landing_title, Modifier.width(231.dp).height(47.dp), ContentScale.Fit)
            Spacer(Modifier.height(24.dp))
            Text(stringResource(R.string.load_tagline), color = Color(0xFFF3ECE4), style = SnType.title)
            Spacer(Modifier.height(8.dp))
            Text(stringResource(R.string.load_sub), color = Color(0xFFD8D2EA), style = SnType.body)
        }
        Column(Modifier.align(Alignment.BottomStart).navigationBarsPadding().padding(start = 28.dp, end = 28.dp, bottom = 40.dp)) {
            words.forEachIndexed { i, w ->
                val lit = progress >= (i + 1) / 4f - 0.01f
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Box(Modifier.size(6.dp).clip(RoundedCornerShape(2.dp)).background(if (lit) Color(0xFFF2C9A1) else Color(0xFF3A3570)))
                    Spacer(Modifier.width(10.dp))
                    Text(stringResource(w), color = if (lit) Color(0xFFF3ECE4) else Color(0xFF6D69A0), fontSize = 11.sp, letterSpacing = 2.4.sp, style = SnType.micro)
                }
                Spacer(Modifier.height(4.dp))
            }
            Spacer(Modifier.height(24.dp))
            if (!ready) {
                Row(Modifier.fillMaxWidth()) {
                    Text(msg, color = Color(0xFFD8D2EA), style = SnType.caption, modifier = Modifier.weight(1f))
                    Text("${(progress * 100).toInt()}%", color = Color(0xFFF2C9A1), style = SnType.caption)
                }
                Spacer(Modifier.height(10.dp))
                Row(Modifier.fillMaxWidth().border(1.dp, Color(0xFF57539A), RoundedCornerShape(10.dp)).padding(5.dp),
                    horizontalArrangement = Arrangement.spacedBy(3.dp)) {
                    repeat(24) { i ->
                        val on = progress * 24 > i
                        Box(Modifier.weight(1f).height(8.dp).clip(RoundedCornerShape(2.dp)).background(if (on) Color(0xFFEAB88C) else Color(0xFF1C1A3A)))
                    }
                }
            } else {
                Box(Modifier.fillMaxWidth().height(54.dp).clip(RoundedCornerShape(18.dp)).background(Color(0xD10A0D1B))
                    .border(1.5.dp, Color(0xFFEAB88C), RoundedCornerShape(18.dp)).clickable(role = Role.Button, onClick = onDone),
                    contentAlignment = Alignment.Center) {
                    Text(stringResource(R.string.load_get_started) + "  →", color = Color(0xFFF4DCC2), fontFamily = Pixelify, fontSize = 17.sp)
                }
            }
            Spacer(Modifier.height(12.dp))
            Text(stringResource(R.string.load_footer), color = Color(0xFFA9A5C6), fontSize = 9.5.sp, letterSpacing = 1.6.sp,
                modifier = Modifier.align(Alignment.CenterHorizontally), style = SnType.micro)
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
        if (page < 2) PrimaryButton(stringResource(R.string.onb_next)) { page++ }
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
            PrimaryButton(stringResource(R.string.setup_import), enabled = st !is ImportState.Copying && st !is ImportState.Verifying) {
                picker.launch(arrayOf("application/octet-stream", "*/*"))
            }
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
