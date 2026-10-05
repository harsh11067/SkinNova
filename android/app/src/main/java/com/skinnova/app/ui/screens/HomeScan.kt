package com.skinnova.app.ui.screens

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Image
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
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.res.stringArrayResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import com.skinnova.app.R
import com.skinnova.app.data.AnalysisEntity
import com.skinnova.app.data.SpotEntity
import com.skinnova.app.ml.Quality
import com.skinnova.app.ml.QualityGate
import com.skinnova.app.ml.toRgb
import com.skinnova.app.model.Tier
import com.skinnova.app.ui.SessionViewModel
import com.skinnova.app.ui.components.OutlineButton
import com.skinnova.app.ui.components.Overline
import com.skinnova.app.ui.components.PrimaryButton
import com.skinnova.app.ui.components.SnCard
import com.skinnova.app.ui.components.SpinRing
import com.skinnova.app.ui.components.StatusPill
import com.skinnova.app.ui.components.TopBar
import com.skinnova.app.ui.components.categoryNameRes
import com.skinnova.app.ui.theme.LocalSn
import com.skinnova.app.ui.theme.SnType
import java.text.DateFormat
import java.util.Date
import java.util.concurrent.Executors

@Composable
fun HomeScreen(vm: SessionViewModel, onScan: () -> Unit, onHistory: () -> Unit, onLibrary: () -> Unit, onSpot: (String) -> Unit,
               onOpenRecent: (AnalysisEntity) -> Unit) {
    val sn = LocalSn.current
    val c = vm.c
    val dark by c.settings.dark.collectAsState()
    val history by c.settings.history.collectAsState()
    val count by c.db.dao().analysisCount().collectAsState(0)
    val recent by c.db.dao().analyses().collectAsState(emptyList())
    val spots by c.db.dao().spots().collectAsState(emptyList())
    val tips = stringArrayResource(R.array.tips)
    var tip by remember { mutableIntStateOf(0) }
    val modelReady = remember { c.models.activeModelPath() != null }
    Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).statusBarsPadding().padding(start = 18.dp, end = 18.dp, top = 18.dp, bottom = 110.dp)) {
        Row(verticalAlignment = Alignment.Top) {
            Column(Modifier.weight(1f)) {
                Text(stringResource(R.string.home_hello), style = SnType.headline, color = sn.ink)
                Spacer(Modifier.height(8.dp))
                Text(stringResource(R.string.home_sub), style = SnType.body, color = sn.mut)
            }
            Box(Modifier.size(48.dp).clip(RoundedCornerShape(16.dp)).background(sn.surf).border(1.dp, sn.line, RoundedCornerShape(16.dp))
                .clickable(role = Role.Button) { c.settings.setDark(!dark) }.semantics { contentDescription = "Toggle light mode" },
                contentAlignment = Alignment.Center) { Text(if (dark) "☀" else "☾", color = sn.accT, fontSize = 18.sp) }
        }
        Spacer(Modifier.height(16.dp))
        StatusPill(stringResource(if (modelReady) R.string.home_status_ready else R.string.home_status_basic), modelReady)
        Spacer(Modifier.height(16.dp))
        SnCard(onClick = onScan) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Box(Modifier.size(76.dp).clip(RoundedCornerShape(20.dp)).background(sn.surf2), contentAlignment = Alignment.Center) {
                    Box(Modifier.size(60.dp).border(1.dp, sn.line, RoundedCornerShape(14.dp)))
                    Text("◉", color = sn.accT, fontSize = 24.sp)
                }
                Spacer(Modifier.width(16.dp))
                Column(Modifier.weight(1f)) {
                    Text(stringResource(R.string.home_scan), style = SnType.title, color = sn.ink)
                    Spacer(Modifier.height(6.dp))
                    Text(stringResource(R.string.home_scan_sub), style = SnType.caption, color = sn.mut)
                }
                Text("›", color = sn.mut, fontSize = 18.sp)
            }
        }
        Spacer(Modifier.height(12.dp))
        Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            SnCard(Modifier.weight(1f), onClick = onHistory) {
                Column {
                    Text("▤", color = sn.ink, fontSize = 18.sp); Spacer(Modifier.height(14.dp))
                    Text(stringResource(R.string.home_reports), style = SnType.label, color = sn.ink); Spacer(Modifier.height(5.dp))
                    Text(if (history) stringResource(R.string.home_reports_sub_n, count) else stringResource(R.string.home_reports_off), style = SnType.caption, color = sn.mut)
                }
            }
            SnCard(Modifier.weight(1f), onClick = onLibrary) {
                Column {
                    Text("✿", color = sn.ink, fontSize = 18.sp); Spacer(Modifier.height(14.dp))
                    Text(stringResource(R.string.home_library), style = SnType.label, color = sn.ink); Spacer(Modifier.height(5.dp))
                    Text(stringResource(R.string.home_library_sub), style = SnType.caption, color = sn.mut)
                }
            }
        }
        Spacer(Modifier.height(12.dp))
        SnCard(onClick = { tip = (tip + 1) % tips.size }) {
            Row {
                Text("◎", color = sn.accT, fontSize = 22.sp); Spacer(Modifier.width(14.dp))
                Column {
                    Row { Text(stringResource(R.string.home_tip), style = SnType.label, color = sn.ink, modifier = Modifier.weight(1f))
                        Text(stringResource(R.string.home_tip_count, tip + 1, tips.size), style = SnType.micro, color = sn.mut) }
                    Spacer(Modifier.height(6.dp))
                    Text(tips[tip], style = SnType.caption, color = sn.mut)
                }
            }
        }
        if (spots.isNotEmpty()) {
            Overline(stringResource(R.string.home_spots))
            spots.forEach { s -> SpotRow(s) { onSpot(s.id) }; Spacer(Modifier.height(8.dp)) }
        }
        recent.firstOrNull()?.let { r ->
            Overline(stringResource(R.string.home_recent))
            SnCard(framed = false, onClick = { onOpenRecent(r) }) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Column(Modifier.weight(1f)) {
                        Text(stringResource(categoryNameRes(r.topKey)), style = SnType.label, color = sn.ink)
                        Text(DateFormat.getDateInstance(DateFormat.MEDIUM).format(Date(r.createdAt)) + " · " +
                            stringResource(tierRes(r.tier)), style = SnType.caption, color = sn.mut)
                    }
                    Text("›", color = sn.mut, fontSize = 18.sp)
                }
            }
        }
    }
}

@Composable
fun SpotRow(s: SpotEntity, onClick: () -> Unit) {
    val sn = LocalSn.current
    val due = Date(System.currentTimeMillis())
    SnCard(framed = false, onClick = onClick) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Box(Modifier.size(40.dp).clip(CircleShape).background(sn.surf2), contentAlignment = Alignment.Center) { Text("◌", color = sn.accT) }
            Spacer(Modifier.width(12.dp))
            Column(Modifier.weight(1f)) {
                Text(s.name, style = SnType.label, color = sn.ink)
                Text(stringResource(R.string.tl_days, s.reminderDays) + " · " + DateFormat.getDateInstance(DateFormat.SHORT).format(due),
                    style = SnType.caption, color = sn.mut)
            }
            Text("›", color = sn.mut, fontSize = 18.sp)
        }
    }
}

fun tierRes(t: String) = when (Tier.parse(t)) {
    Tier.URGENT -> R.string.tier_URGENT; Tier.HIGH -> R.string.tier_HIGH; Tier.MODERATE -> R.string.tier_MODERATE; else -> R.string.tier_LOW
}

/** 03 Scan (design): framed viewfinder with dashed circle, live hint pill, Photo/Gallery switch, shutter. */
@Composable
fun ScanScreen(vm: SessionViewModel, onBack: () -> Unit, onPhotoAccepted: () -> Unit) {
    val sn = LocalSn.current
    val ctx = LocalContext.current
    var hasCam by remember { mutableStateOf(ContextCompat.checkSelfPermission(ctx, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) }
    val camPerm = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { hasCam = it }
    var gallery by remember { mutableStateOf(false) }
    var hint by remember { mutableStateOf<Quality?>(null) }
    var torch by remember { mutableStateOf(false) }
    val capture = remember { ImageCapture.Builder().setCaptureMode(ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY).build() }
    var camera by remember { mutableStateOf<androidx.camera.core.Camera?>(null) }
    val photo by vm.photo.collectAsState()
    val quality by vm.quality.collectAsState()
    var reviewing by remember { mutableStateOf(false) }
    val picker = rememberLauncherForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
        if (uri != null) { vm.setPhotoFromUri(uri); reviewing = true }
    }
    LaunchedEffect(Unit) { if (!hasCam) camPerm.launch(Manifest.permission.CAMERA) }

    Column(Modifier.fillMaxSize().statusBarsPadding().navigationBarsPadding().padding(horizontal = 18.dp, vertical = 8.dp)) {
        TopBar(stringResource(R.string.scan_title), onBack)
        Text(stringResource(R.string.scan_sub), style = SnType.body, color = sn.mut, modifier = Modifier.fillMaxWidth().padding(horizontal = 8.dp))
        Spacer(Modifier.height(14.dp))
        Box(Modifier.fillMaxWidth().height(330.dp).clip(RoundedCornerShape(26.dp)).border(1.5.dp, sn.acc, RoundedCornerShape(26.dp)).padding(5.dp)
            .clip(RoundedCornerShape(21.dp)).background(Color.Black)) {
            val p = photo
            if (reviewing && p != null) {
                Image(p.asImageBitmap(), null, Modifier.fillMaxSize(), contentScale = ContentScale.Crop)
            } else if (hasCam && !gallery) {
                CameraPreview(capture, onFrame = { hint = it }, onCamera = { camera = it })
                Box(Modifier.align(Alignment.Center).size(150.dp)) { SpinRing(150.dp, Color(0xBFFFF0E1), 24000, dotted = false) }
            } else {
                Column(Modifier.align(Alignment.Center).padding(24.dp), horizontalAlignment = Alignment.CenterHorizontally) {
                    Text(stringResource(R.string.scan_no_camera), style = SnType.body, color = Color(0xFFF3ECE4))
                    if (!hasCam) { Spacer(Modifier.height(10.dp)); OutlineButton(stringResource(R.string.scan_allow_camera)) { camPerm.launch(Manifest.permission.CAMERA) } }
                }
            }
            val h = if (reviewing) quality else hint
            if (h != null) {
                val (txt, ok) = hintText(h)
                Row(Modifier.align(Alignment.BottomStart).padding(13.dp).height(28.dp).clip(RoundedCornerShape(14.dp)).background(Color(0xA6060A17))
                    .padding(horizontal = 12.dp), verticalAlignment = Alignment.CenterVertically) {
                    Box(Modifier.size(6.dp).clip(CircleShape).background(if (ok) Color(0xFF6FD39A) else Color(0xFFF2C46D)))
                    Spacer(Modifier.width(6.dp)); Text(txt, fontSize = 10.sp, color = Color(0xFFF3ECE4), style = SnType.micro)
                }
            }
        }
        Spacer(Modifier.height(18.dp))
        if (!reviewing) {
            Row(Modifier.align(Alignment.CenterHorizontally).width(200.dp).height(44.dp).clip(RoundedCornerShape(20.dp)).background(sn.surf)
                .border(1.dp, sn.line2, RoundedCornerShape(20.dp)).padding(4.dp)) {
                listOf(false to R.string.scan_mode_photo, true to R.string.scan_mode_gallery).forEach { (g, l) ->
                    Box(Modifier.weight(1f).fillMaxSize().clip(RoundedCornerShape(16.dp)).background(if (gallery == g) sn.surf2 else Color.Transparent)
                        .border(1.dp, if (gallery == g) sn.acc else Color.Transparent, RoundedCornerShape(16.dp))
                        .clickable(role = Role.Tab) { gallery = g; if (g) picker.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)) },
                        contentAlignment = Alignment.Center) { Text(stringResource(l), style = SnType.bodyL, color = sn.ink) }
                }
            }
            Spacer(Modifier.weight(1f))
            Row(Modifier.fillMaxWidth().padding(horizontal = 18.dp), verticalAlignment = Alignment.CenterVertically) {
                RoundIcon("▣", stringResource(R.string.scan_pick)) { picker.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)) }
                Spacer(Modifier.weight(1f))
                Box(Modifier.size(88.dp).clickable(role = Role.Button, enabled = hasCam && !gallery) {
                    capture.takePicture(ContextCompat.getMainExecutor(ctx), object : ImageCapture.OnImageCapturedCallback() {
                        override fun onCaptureSuccess(image: ImageProxy) {
                            val bmp = image.toBitmap().rotate(image.imageInfo.rotationDegrees); image.close()
                            vm.setPhoto(bmp); reviewing = true
                        }
                        override fun onError(exception: ImageCaptureException) {}
                    })
                }.semantics { contentDescription = "Take photo" }, contentAlignment = Alignment.Center) {
                    SpinRing(88.dp, sn.acc, 14000)
                    Box(Modifier.size(70.dp).clip(CircleShape).border(1.5.dp, sn.acc, CircleShape))
                    Box(Modifier.size(58.dp).clip(CircleShape).background(Color(0xFFF5DDC2)), contentAlignment = Alignment.Center) { Text("◉", color = Color(0xFF5A3A22), fontSize = 18.sp) }
                }
                Spacer(Modifier.weight(1f))
                RoundIcon(if (torch) "ϟ" else "↯", stringResource(R.string.scan_torch)) { torch = !torch; camera?.cameraControl?.enableTorch(torch) }
            }
            Spacer(Modifier.height(12.dp))
        } else {
            val q = quality
            Spacer(Modifier.weight(1f))
            if (q != null && !q.ok) {
                SnCard(border = sn.moderate, framed = false) {
                    Column {
                        Text(stringResource(R.string.qg_title), style = SnType.title, color = sn.moderate)
                        q.issues.forEach { Text("• " + stringResource(issueRes(it)), style = SnType.body, color = sn.ink) }
                    }
                }
                Spacer(Modifier.height(12.dp))
                Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                    OutlineButton(stringResource(R.string.qg_retake), Modifier.weight(1f)) { reviewing = false }
                    OutlineButton(stringResource(R.string.qg_use_anyway), Modifier.weight(1f)) { vm.forceQuality(); onPhotoAccepted() }
                }
            } else if (q != null) {
                Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                    OutlineButton(stringResource(R.string.qg_retake), Modifier.weight(1f)) { reviewing = false }
                    PrimaryButton(stringResource(R.string.q_next), Modifier.weight(1f)) { onPhotoAccepted() }
                }
            }
            Spacer(Modifier.height(12.dp))
        }
    }
}

@Composable
private fun RoundIcon(glyph: String, label: String, onClick: () -> Unit) {
    val sn = LocalSn.current
    Box(Modifier.size(50.dp).clip(RoundedCornerShape(18.dp)).background(sn.surf).border(1.dp, sn.line, RoundedCornerShape(18.dp))
        .clickable(role = Role.Button, onClick = onClick).semantics { contentDescription = label }, contentAlignment = Alignment.Center) {
        Text(glyph, color = sn.ink, fontSize = 18.sp)
    }
}

@Composable
fun hintText(q: Quality): Pair<String, Boolean> = when {
    Quality.Issue.DARK in q.issues -> stringResource(R.string.scan_hint_dark) to false
    Quality.Issue.BRIGHT in q.issues -> stringResource(R.string.scan_hint_bright) to false
    Quality.Issue.BLURRY in q.issues -> stringResource(R.string.scan_hint_blur) to false
    Quality.Issue.NO_SKIN in q.issues -> stringResource(R.string.scan_hint_close) to false
    else -> stringResource(R.string.scan_hint_ok) to true
}

fun issueRes(i: Quality.Issue) = when (i) {
    Quality.Issue.BLURRY -> R.string.qg_blurry; Quality.Issue.DARK -> R.string.qg_dark; Quality.Issue.BRIGHT -> R.string.qg_bright
    Quality.Issue.SMALL -> R.string.qg_small; Quality.Issue.NO_SKIN -> R.string.qg_noskin
}

fun Bitmap.rotate(deg: Int): Bitmap = if (deg == 0) this else Bitmap.createBitmap(this, 0, 0, width, height, Matrix().apply { postRotate(deg.toFloat()) }, true)

/** CameraX preview + capture + 2 Hz analysis frames for live quality hints (and the timeline align score). */
@Composable
fun CameraPreview(capture: ImageCapture, onFrame: (Quality) -> Unit, onCamera: (androidx.camera.core.Camera) -> Unit = {},
                  onBitmap: ((Bitmap) -> Unit)? = null) {
    val ctx = LocalContext.current
    val owner = LocalLifecycleOwner.current
    val exec = remember { Executors.newSingleThreadExecutor() }
    DisposableEffect(Unit) { onDispose { exec.shutdown() } }
    AndroidView(factory = { c ->
        val view = PreviewView(c).apply { scaleType = PreviewView.ScaleType.FILL_CENTER }
        val providerF = ProcessCameraProvider.getInstance(c)
        providerF.addListener({
            val provider = providerF.get()
            val preview = Preview.Builder().build().also { it.surfaceProvider = view.surfaceProvider }
            var last = 0L
            val analysis = ImageAnalysis.Builder().setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888).build().also {
                    it.setAnalyzer(exec) { img ->
                        val now = System.currentTimeMillis()
                        if (now - last > 500) {
                            last = now
                            val b = img.toBitmap().rotate(img.imageInfo.rotationDegrees)
                            val small = SessionViewModel.downscale(b, 320)
                            val q = QualityGate.check(small.toRgb())
                            ContextCompat.getMainExecutor(ctx).execute { onFrame(q) }
                            onBitmap?.invoke(small)
                        }
                        img.close()
                    }
                }
            runCatching {
                provider.unbindAll()
                val cam = provider.bindToLifecycle(owner, CameraSelector.DEFAULT_BACK_CAMERA, preview, capture, analysis)
                onCamera(cam)
            }
        }, ContextCompat.getMainExecutor(c))
        view
    }, modifier = Modifier.fillMaxSize())
}

