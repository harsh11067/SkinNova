package com.skinnova.app.ui.screens

import android.content.Intent
import android.graphics.Bitmap
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
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
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.unit.dp
import androidx.core.content.FileProvider
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.core.content.ContextCompat
import com.skinnova.app.R
import com.skinnova.app.data.CaptureEntity
import com.skinnova.app.data.MetricsEntity
import com.skinnova.app.model.CvScore
import com.skinnova.app.model.SnJson
import com.skinnova.app.model.Tier
import com.skinnova.app.reminders.ReminderWorker
import com.skinnova.app.report.DoctorSummaryPdf
import com.skinnova.app.timeline.ChangeMetrics
import com.skinnova.app.timeline.Timeline
import com.skinnova.app.ui.RecaptureState
import com.skinnova.app.ui.SessionViewModel
import com.skinnova.app.ui.TimelineViewModel
import com.skinnova.app.ui.components.Chip
import com.skinnova.app.ui.components.Disclaimer
import com.skinnova.app.ui.components.OutlineButton
import com.skinnova.app.ui.components.PrimaryButton
import com.skinnova.app.ui.components.SmallTag
import com.skinnova.app.ui.components.SnCard
import com.skinnova.app.ui.components.TopBar
import com.skinnova.app.ui.components.categoryNameRes
import com.skinnova.app.ui.theme.LocalSn
import com.skinnova.app.ui.theme.SnType
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlinx.serialization.builtins.ListSerializer
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import java.text.DateFormat
import java.util.Date

@Composable
fun TrackSpotScreen(vm: SessionViewModel, tvm: TimelineViewModel, onBack: () -> Unit, onCreated: (String) -> Unit) {
    val sn = LocalSn.current
    val photo by vm.photo.collectAsState()
    val cv by vm.cv.collectAsState()
    val res by vm.result.collectAsState()
    // pre-filled ("Arm spot") so "Start tracking" works in one tap; the user can rename it
    val defaultName = stringResource(R.string.tl_default_name, stringResource(com.skinnova.app.ui.components.siteNameRes(res?.answers?.bodySite)))
    var name by remember { mutableStateOf(defaultName) }
    var days by remember { mutableIntStateOf(14) }
    var seed by remember { mutableStateOf(Offset(0.5f, 0.5f)) }
    var mask by remember { mutableStateOf<Bitmap?>(null) }
    val scope = rememberCoroutineScope()
    val p = photo ?: return
    LaunchedEffect(seed) { mask = withContext(Dispatchers.Default) { runCatching { Timeline.maskBitmap(p, seed.x.toDouble(), seed.y.toDouble()) }.getOrNull() } }
    Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).statusBarsPadding().navigationBarsPadding().padding(horizontal = 18.dp, vertical = 8.dp)) {
        TopBar(stringResource(R.string.tl_track_title), onBack)
        Text(stringResource(R.string.tl_tap_seed), style = SnType.body, color = sn.mut)
        Spacer(Modifier.height(8.dp))
        Box(Modifier.fillMaxWidth().aspectRatio(p.width.toFloat() / p.height).clip(RoundedCornerShape(22.dp)).border(1.5.dp, sn.acc, RoundedCornerShape(22.dp))
            .pointerInput(Unit) { detectTapGestures { o -> seed = Offset(o.x / size.width, o.y / size.height) } }) {
            Image(p.asImageBitmap(), null, Modifier.fillMaxSize(), contentScale = ContentScale.FillBounds)
            mask?.let { Image(it.asImageBitmap(), null, Modifier.fillMaxSize().alpha(0.35f), contentScale = ContentScale.FillBounds) }
            Canvas(Modifier.fillMaxSize()) { drawCircle(Color(0xFFEAB88C), 14f, Offset(seed.x * size.width, seed.y * size.height), style = Stroke(4f)) }
        }
        Spacer(Modifier.height(16.dp))
        Text(stringResource(R.string.tl_name), style = SnType.label, color = sn.ink)
        Spacer(Modifier.height(6.dp))
        BasicTextField(name, { name = it.take(40) }, Modifier.fillMaxWidth().clip(RoundedCornerShape(16.dp)).background(sn.surf).border(1.dp, sn.line, RoundedCornerShape(16.dp)).padding(horizontal = 14.dp),
            textStyle = com.skinnova.app.ui.components.fieldTextStyle.copy(color = sn.ink), cursorBrush = SolidColor(sn.acc), singleLine = true,
            decorationBox = { inner -> if (name.isEmpty()) Text(stringResource(R.string.tl_name_hint), style = com.skinnova.app.ui.components.fieldTextStyle, color = sn.mut); inner() })
        Spacer(Modifier.height(16.dp))
        Text(stringResource(R.string.tl_reminder), style = SnType.label, color = sn.ink)
        Spacer(Modifier.height(8.dp))
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) { listOf(3, 7, 14, 30).forEach { d -> Chip(stringResource(R.string.tl_days, d), days == d, { days = d }) } }
        Spacer(Modifier.height(16.dp))
        Box(Modifier.fillMaxWidth().clip(RoundedCornerShape(20.dp)).background(sn.surf2).padding(14.dp)) { Text("◎  " + stringResource(R.string.tl_coin_tip), style = SnType.caption, color = sn.mut) }
        Spacer(Modifier.height(18.dp))
        PrimaryButton(stringResource(R.string.tl_start), enabled = name.isNotBlank()) {
            scope.launch { tvm.createSpot(name.trim(), res?.answers?.bodySite ?: "other", seed.x, seed.y, days, p, cv, vm.savedId, onCreated) }
        }
        Spacer(Modifier.height(24.dp))
    }
}

@Composable
fun TimelineScreen(vm: SessionViewModel, tvm: TimelineViewModel, spotId: String, onBack: () -> Unit, onRecapture: (String) -> Unit) {
    val sn = LocalSn.current
    val ctx = LocalContext.current
    val c = vm.c
    val scope = rememberCoroutineScope()
    val caps by c.db.dao().captures(spotId).collectAsState(emptyList())
    val mets by c.db.dao().metrics(spotId).collectAsState(emptyList())
    var spot by remember { mutableStateOf<com.skinnova.app.data.SpotEntity?>(null) }
    val thumbs = remember { mutableStateOf<Map<String, Bitmap>>(emptyMap()) }
    LaunchedEffect(spotId, caps.size) {
        spot = c.db.dao().spot(spotId)
        thumbs.value = caps.associate { it.id to (c.images.load(it.imageRef)?.let { b -> SessionViewModel.downscale(b, 240) }) }.filterValues { it != null }.mapValues { it.value!! }
    }
    val s = spot ?: return
    val metrics = mets.associateBy { it.captureId }
    val parsed = mets.mapNotNull { runCatching { SnJson.decodeFromString(ChangeMetrics.serializer(), it.json) }.getOrNull() }.sortedBy { it.daysSinceBaseline }
    val latest = caps.lastOrNull()?.let { metrics[it.id] }
    Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).statusBarsPadding().navigationBarsPadding().padding(horizontal = 18.dp, vertical = 8.dp)) {
        TopBar(s.name, onBack)
        Text(stringResource(R.string.tl_captures, caps.size) + " · " + stringResource(R.string.tl_days, s.reminderDays), style = SnType.caption, color = sn.mut)
        Spacer(Modifier.height(12.dp))
        Row(Modifier.horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(10.dp)) {
            caps.forEach { cap ->
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    thumbs.value[cap.id]?.let { Image(it.asImageBitmap(), null, Modifier.size(84.dp).clip(RoundedCornerShape(16.dp))
                        .border(if (cap.id == s.baselineCaptureId) 1.5.dp else 0.dp, sn.acc, RoundedCornerShape(16.dp)), contentScale = ContentScale.Crop) }
                    Text(DateFormat.getDateInstance(DateFormat.SHORT).format(Date(cap.takenAt)), style = SnType.micro, color = sn.mut)
                    if (cap.id == s.baselineCaptureId) Text(stringResource(R.string.tl_baseline), style = SnType.micro, color = sn.accT)
                    metrics[cap.id]?.let { m -> Text(stringResource(tierRes(m.timelineTier)), style = SnType.micro, color = sn.tier(Tier.parse(m.timelineTier) ?: Tier.LOW)) }
                }
            }
        }
        Spacer(Modifier.height(14.dp))
        if (parsed.isNotEmpty()) {
            SnCard(framed = false) {
                Column {
                    Text(stringResource(R.string.tl_size), style = SnType.label, color = sn.ink)
                    Sparkline(listOf(1.0) + parsed.map { it.areaRatio ?: 1.0 }, sn.acc)
                    Spacer(Modifier.height(10.dp))
                    Text(stringResource(R.string.tl_contrast), style = SnType.label, color = sn.ink)
                    Sparkline(listOf(0.0) + parsed.map { it.contrastDelta ?: 0.0 }, sn.pur)
                    val last = parsed.last()
                    Spacer(Modifier.height(10.dp))
                    Text(stringResource(if (last.confidence == "ok") R.string.tl_conf_ok else R.string.tl_conf_low), style = SnType.label,
                        color = if (last.confidence == "ok") sn.low else sn.moderate)
                    if (last.confidence == "low") {
                        val why = listOfNotNull(if (!last.coinInBoth) stringResource(R.string.tl_why_coin) else null,
                            if (last.coinInBoth && (last.coinScaleErr ?: 0.0) > 0.10) stringResource(R.string.tl_why_scale) else null,
                            if (!last.segOk) stringResource(R.string.tl_why_seg) else null,
                            if ((last.noiseFloor?.n ?: 0) < 3) stringResource(R.string.tl_why_noise) else null).joinToString(", ")
                        if (why.isNotEmpty()) Text(stringResource(R.string.tl_conf_low_why, why), style = SnType.caption, color = sn.mut)
                    }
                }
            }
            Spacer(Modifier.height(12.dp))
        }
        latest?.narration?.let { n ->
            SnCard(framed = false) { Column { Text(stringResource(R.string.tl_summary), style = SnType.title, color = sn.ink); Spacer(Modifier.height(8.dp)); Text(n, style = SnType.bodyL, color = sn.ink) } }
            Spacer(Modifier.height(12.dp))
        }
        latest?.let { m -> val t = Tier.parse(m.timelineTier) ?: Tier.LOW
            if (t >= Tier.MODERATE) { com.skinnova.app.ui.components.TriageCard(t); Spacer(Modifier.height(12.dp)) } }
        PrimaryButton(stringResource(R.string.tl_recapture)) { onRecapture(spotId) }
        Spacer(Modifier.height(10.dp))
        if (s.noiseN < 3) {
            OutlineButton(stringResource(R.string.tl_calibrate), Modifier.fillMaxWidth()) { onRecapture("$spotId?cal") }
            Text(stringResource(R.string.tl_calibrate_sub), style = SnType.micro, color = sn.mut, modifier = Modifier.padding(6.dp))
            Spacer(Modifier.height(6.dp))
        }
        OutlineButton(stringResource(R.string.tl_export), Modifier.fillMaxWidth()) {
            scope.launch {
                val f = withContext(Dispatchers.IO) { exportPdf(vm, s, caps, metrics) }
                val uri = FileProvider.getUriForFile(ctx, ctx.packageName + ".files", f)
                ctx.startActivity(Intent.createChooser(Intent(Intent.ACTION_SEND).setType("application/pdf").putExtra(Intent.EXTRA_STREAM, uri)
                    .addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION), null))
            }
        }
        Spacer(Modifier.height(10.dp))
        Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
            caps.lastOrNull()?.takeIf { it.id != s.baselineCaptureId }?.let { last ->
                OutlineButton(stringResource(R.string.tl_rebaseline), Modifier.weight(1f)) { scope.launch { c.db.dao().put(s.copy(baselineCaptureId = last.id)); spot = c.db.dao().spot(spotId) } }
            }
            OutlineButton(stringResource(R.string.tl_delete), Modifier.weight(1f)) {
                scope.launch { ReminderWorker.cancel(ctx, spotId); caps.forEach { c.images.delete(it.imageRef) }; c.db.dao().deleteCaptures(spotId); c.db.dao().deleteSpot(spotId); onBack() }
            }
        }
        Spacer(Modifier.height(12.dp))
        Disclaimer()
        Spacer(Modifier.height(24.dp))
    }
}

private suspend fun exportPdf(vm: SessionViewModel, s: com.skinnova.app.data.SpotEntity, caps: List<CaptureEntity>, metrics: Map<String, MetricsEntity>): java.io.File {
    val c = vm.c
    val rows = caps.map { cap ->
        val cv = runCatching { SnJson.decodeFromString(ListSerializer(CvScore.serializer()), cap.cvProbsJson) }.getOrDefault(emptyList())
        val m = metrics[cap.id]
        DoctorSummaryPdf.Row(cap.takenAt, c.images.load(cap.imageRef)?.let { SessionViewModel.downscale(it, 300) },
            cv.take(3).joinToString(", ") { vm.getApplication<android.app.Application>().getString(categoryNameRes(it.key)) + " " + "%.2f".format(it.p) },
            m?.timelineTier ?: "—", m?.let { runCatching { SnJson.decodeFromString(ChangeMetrics.serializer(), it.json) }.getOrNull() }, m?.narration)
    }
    val res = vm.result.value
    val app = vm.getApplication<android.app.Application>()
    val top = res?.output?.possibleCategories?.firstOrNull()?.key
    val questions = top?.let { k -> ((c.prompts.cards[k] as? JsonObject)?.get("see_doctor_if") as? JsonArray)?.mapNotNull { (it as? JsonPrimitive)?.content?.let { q -> "Should I worry if $q?" } } } ?: emptyList()
    val answers = res?.answers?.let { "${it.bodySite}, ${it.duration}, itch ${it.itch}/3, pain ${it.pain}/3, changing: ${it.changing}" } ?: "—"
    val msgs = res?.ruleMessages?.mapNotNull { com.skinnova.app.ui.components.ruleMessageRes(it)?.let { r -> if (it.startsWith("rf_t")) app.getString(r, "") else app.getString(r) } } ?: emptyList()
    return DoctorSummaryPdf.build(app, s, rows, answers, msgs, questions, c.models.activeModel()?.sha256 ?: "")
}

@Composable
private fun Sparkline(values: List<Double>, color: Color) {
    val sn = LocalSn.current
    Canvas(Modifier.fillMaxWidth().height(48.dp).padding(vertical = 6.dp)) {
        if (values.size < 2) return@Canvas
        val lo = values.min(); val hi = values.max(); val span = (hi - lo).takeIf { it > 1e-6 } ?: 1.0
        val pts = values.mapIndexed { i, v -> Offset(i * size.width / (values.size - 1), (size.height - ((v - lo) / span * size.height)).toFloat()) }
        for (i in 1 until pts.size) drawLine(color, pts[i - 1], pts[i], 4f)
        pts.forEach { drawCircle(sn.ink, 5f, it) }
    }
}

/** Re-capture with ghost overlay (35 % previous photo) and a live alignment ring (ORB inlier ratio, every 500 ms). */
@Composable
fun RecaptureScreen(vm: SessionViewModel, tvm: TimelineViewModel, spotArg: String, onDone: () -> Unit) {
    val sn = LocalSn.current
    val ctx = LocalContext.current
    val calibration = spotArg.endsWith("?cal")
    val spotId = spotArg.removeSuffix("?cal")
    val baseline by tvm.baseline.collectAsState()
    val seed by tvm.seed.collectAsState()
    val st by tvm.rec.collectAsState()
    var align by remember { mutableFloatStateOf(0f) }
    var coin by remember { mutableStateOf(false) }
    val capture = remember { ImageCapture.Builder().setCaptureMode(ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY).build() }
    LaunchedEffect(spotId) { tvm.reset(); tvm.loadBaseline(spotId) }
    val baseSmall = remember(baseline) { baseline?.let { runCatching { Timeline.resizeMax(Timeline.toRgbMat(SessionViewModel.downscale(it, 320)), 320) }.getOrNull() } }
    Column(Modifier.fillMaxSize().statusBarsPadding().navigationBarsPadding().padding(horizontal = 18.dp, vertical = 8.dp)) {
        TopBar(stringResource(if (calibration) R.string.tl_calibrate else R.string.tl_recapture), { tvm.reset(); onDone() })
        Text(stringResource(R.string.tl_align), style = SnType.body, color = sn.mut)
        Spacer(Modifier.height(10.dp))
        val ring = when { align >= 0.35f -> sn.low; align >= 0.2f -> sn.moderate; else -> sn.urgent }
        Box(Modifier.fillMaxWidth().height(380.dp).clip(RoundedCornerShape(26.dp)).border(3.dp, ring, RoundedCornerShape(26.dp)).background(Color.Black)) {
            if (st is RecaptureState.Idle) {
                CameraPreview(capture, onFrame = {}, onBitmap = { small ->
                    val b = baseSmall ?: return@CameraPreview
                    runCatching {
                        val m = Timeline.toRgbMat(small); val a = Timeline.align(b, m)
                        align = if (a.h == null) 0f else a.ratio.toFloat()
                        val sp = seed?.let { (x, y) -> a.h?.let { h -> Timeline.seedInNew(h, x, y, b) } }   // the spot in this frame
                        coin = Timeline.detectCoin(m, sp) != null
                    }
                })
                baseline?.let { Image(it.asImageBitmap(), null, Modifier.fillMaxSize().alpha(0.35f), contentScale = ContentScale.Crop) }
            }
            Row(Modifier.align(Alignment.TopStart).padding(12.dp), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                SmallTag(stringResource(when { align >= 0.35f -> R.string.tl_align_ok; align >= 0.2f -> R.string.tl_align_weak; else -> R.string.tl_align_bad }), ring)
                if (coin) SmallTag("◎ " + stringResource(R.string.tl_coin), sn.low)
            }
        }
        Spacer(Modifier.height(16.dp))
        when (val s = st) {
            RecaptureState.Idle -> PrimaryButton(stringResource(R.string.scan_shoot)) {
                capture.takePicture(ContextCompat.getMainExecutor(ctx), object : ImageCapture.OnImageCapturedCallback() {
                    override fun onCaptureSuccess(image: ImageProxy) {
                        val bmp = image.toBitmap().rotate(image.imageInfo.rotationDegrees); image.close()
                        tvm.recapture(spotId, SessionViewModel.downscale(bmp, 1024), calibration)
                    }
                    override fun onError(exception: ImageCaptureException) {}
                })
            }
            RecaptureState.Measuring, RecaptureState.Narrating -> Text(stringResource(R.string.an_title) + "…", style = SnType.label, color = sn.ink)
            RecaptureState.NoMatch -> {
                Text(stringResource(R.string.tl_no_match), style = SnType.body, color = sn.moderate)
                Spacer(Modifier.height(10.dp)); OutlineButton(stringResource(R.string.qg_retake), Modifier.fillMaxWidth()) { tvm.reset() }
            }
            is RecaptureState.Done -> PrimaryButton(stringResource(R.string.v_confirm)) { tvm.reset(); onDone() }
            is RecaptureState.Error -> { Text(s.msg, style = SnType.caption, color = sn.urgent); OutlineButton(stringResource(R.string.qg_retake), Modifier.fillMaxWidth()) { tvm.reset() } }
        }
    }
}

