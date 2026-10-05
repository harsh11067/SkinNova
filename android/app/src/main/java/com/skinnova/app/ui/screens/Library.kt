package com.skinnova.app.ui.screens

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
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.skinnova.app.BuildConfig
import com.skinnova.app.R
import com.skinnova.app.data.AnalysisEntity
import com.skinnova.app.ml.CvPreprocess
import com.skinnova.app.ui.SessionViewModel
import com.skinnova.app.ui.components.Chip
import com.skinnova.app.ui.components.Disclaimer
import com.skinnova.app.ui.components.OutlineButton
import com.skinnova.app.ui.components.Overline
import com.skinnova.app.ui.components.PixelImage
import com.skinnova.app.ui.components.SnCard
import com.skinnova.app.ui.components.Toggle
import com.skinnova.app.ui.components.TopBar
import com.skinnova.app.ui.components.categoryArt
import com.skinnova.app.ui.components.categoryNameRes
import com.skinnova.app.ui.theme.LocalSn
import com.skinnova.app.ui.theme.SnType
import kotlinx.coroutines.launch
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import java.text.DateFormat
import java.util.Calendar
import java.util.Date

private fun JsonObject.strs(k: String) = (this[k] as? JsonArray)?.mapNotNull { (it as? JsonPrimitive)?.content } ?: emptyList()
private fun JsonObject.str(k: String) = (this[k] as? JsonPrimitive)?.content ?: ""

/** 07 Detailed Insights: About / Care / See a doctor tabs, from condition_cards.json (same notes the LLM sees). */
@Composable
fun InsightsScreen(vm: SessionViewModel, key: String, onBack: () -> Unit) {
    val sn = LocalSn.current
    val card = vm.c.prompts.cards[key] as? JsonObject ?: return
    var tab by remember { mutableIntStateOf(0) }
    Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).statusBarsPadding().navigationBarsPadding().padding(horizontal = 18.dp, vertical = 8.dp)) {
        TopBar(stringResource(R.string.ins_title), onBack)
        Row(Modifier.fillMaxWidth().clip(RoundedCornerShape(20.dp)).background(sn.surf).border(1.dp, sn.line2, RoundedCornerShape(20.dp)).padding(4.dp),
            horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            listOf(R.string.ins_tab_about, R.string.ins_tab_care, R.string.ins_tab_doctor).forEachIndexed { i, l ->
                Box(Modifier.weight(1f).heightIn(min = 44.dp).clip(RoundedCornerShape(16.dp)).background(if (tab == i) sn.surf2 else Color.Transparent)
                    .border(1.dp, if (tab == i) sn.acc else Color.Transparent, RoundedCornerShape(16.dp)).clickable(role = Role.Tab) { tab = i },
                    contentAlignment = Alignment.Center) { Text(stringResource(l), style = SnType.body, color = if (tab == i) sn.accT else sn.ink) }
            }
        }
        Spacer(Modifier.height(14.dp))
        SnCard {
            Column {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    PixelImage(categoryArt(key), Modifier.size(64.dp).clip(RoundedCornerShape(18.dp)).border(1.5.dp, sn.acc, RoundedCornerShape(18.dp)))
                    Spacer(Modifier.width(14.dp))
                    Text(stringResource(categoryNameRes(key)), style = SnType.title, color = sn.ink)
                }
                Spacer(Modifier.height(10.dp))
                Text(card.str("summary"), style = SnType.body, color = sn.mut)
            }
        }
        Spacer(Modifier.height(12.dp))
        val (title, items) = when (tab) {
            0 -> stringResource(R.string.ins_signs) to card.strs("typical_features")
            1 -> stringResource(R.string.ins_care) to card.strs("general_care")
            else -> stringResource(R.string.ins_see) to card.strs("see_doctor_if")
        }
        SnCard(framed = false) {
            Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                Text(title, style = SnType.title, color = sn.ink)
                items.forEach { t ->
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Box(Modifier.size(24.dp).clip(RoundedCornerShape(9.dp)).border(1.dp, sn.line, RoundedCornerShape(9.dp)), contentAlignment = Alignment.Center) {
                            Text(if (tab == 2) "!" else "✓", color = sn.accT, fontSize = 10.sp)
                        }
                        Spacer(Modifier.width(12.dp))
                        Text(t.replaceFirstChar { it.uppercase() }, style = SnType.body, color = sn.ink)
                    }
                }
            }
        }
        Spacer(Modifier.height(12.dp))
        Text(stringResource(R.string.ins_sources) + ": " + card.strs("sources").joinToString(" · "), style = SnType.micro, color = sn.mut)
        Spacer(Modifier.height(12.dp))
        PixelImage(R.drawable.px_insights, Modifier.fillMaxWidth().height(96.dp).clip(RoundedCornerShape(24.dp)))
        Spacer(Modifier.height(12.dp))
        Disclaimer()
        Spacer(Modifier.height(24.dp))
    }
}

/** 08 Skin Library: offline condition notes with filter chips. */
@Composable
fun LibraryScreen(vm: SessionViewModel, onOpen: (String) -> Unit) {
    val sn = LocalSn.current
    var filter by remember { mutableIntStateOf(0) }
    val groups = listOf(
        vm.c.labels.keys,
        listOf("eczema_atopic", "contact_dermatitis", "seborrheic_dermatitis", "psoriasis", "acne", "vitiligo"),
        listOf("tinea", "scabies"),
        listOf("benign_lesion", "suspicious_lesion", "other"),
    )
    Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).statusBarsPadding().padding(start = 18.dp, end = 18.dp, top = 18.dp, bottom = 110.dp)) {
        Text(stringResource(R.string.lib_title), style = SnType.headline, color = sn.ink)
        Spacer(Modifier.height(6.dp))
        Text(stringResource(R.string.lib_sub), style = SnType.body, color = sn.mut)
        Spacer(Modifier.height(16.dp))
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            listOf(R.string.lib_all, R.string.lib_rash, R.string.lib_infection, R.string.lib_spots).forEachIndexed { i, l -> Chip(stringResource(l), filter == i, { filter = i }) }
        }
        Spacer(Modifier.height(16.dp))
        groups[filter].forEach { k ->
            val card = vm.c.prompts.cards[k] as? JsonObject
            SnCard(onClick = { onOpen(k) }) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    PixelImage(categoryArt(k), Modifier.size(58.dp).clip(RoundedCornerShape(16.dp)))
                    Spacer(Modifier.width(14.dp))
                    Column(Modifier.weight(1f)) {
                        Text(stringResource(categoryNameRes(k)), style = SnType.label, color = sn.ink)
                        Spacer(Modifier.height(4.dp))
                        Text(card?.str("summary")?.substringBefore(". ")?.plus(".") ?: "", style = SnType.caption, color = sn.mut, maxLines = 2)
                    }
                    Text("›", color = sn.mut, fontSize = 18.sp)
                }
            }
            Spacer(Modifier.height(10.dp))
        }
        Box(Modifier.fillMaxWidth().height(110.dp).clip(RoundedCornerShape(24.dp))) {
            PixelImage(R.drawable.px_library, Modifier.fillMaxSize())
            Text(stringResource(R.string.lib_quote), style = SnType.caption, color = Color(0xFFECE4F2), modifier = Modifier.padding(start = 66.dp, top = 22.dp))
        }
    }
}

/** 09 Your Journey: day strip, saved analyses (opt-in), tracked spots. */
@Composable
fun HistoryScreen(vm: SessionViewModel, onOpen: (AnalysisEntity) -> Unit, onSpot: (String) -> Unit, onScan: () -> Unit) {
    val sn = LocalSn.current
    val history by vm.c.settings.history.collectAsState()
    val items by vm.c.db.dao().analyses().collectAsState(emptyList())
    val spots by vm.c.db.dao().spots().collectAsState(emptyList())
    val days = remember { (4 downTo 0).map { Calendar.getInstance().apply { add(Calendar.DAY_OF_YEAR, -it) } } }
    var day by remember { mutableStateOf<Int?>(null) }
    fun sameDay(t: Long, c: Calendar) = Calendar.getInstance().apply { timeInMillis = t }.let { it.get(Calendar.YEAR) == c.get(Calendar.YEAR) && it.get(Calendar.DAY_OF_YEAR) == c.get(Calendar.DAY_OF_YEAR) }
    Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).statusBarsPadding().padding(start = 18.dp, end = 18.dp, top = 18.dp, bottom = 110.dp)) {
        Text(stringResource(R.string.hist_title), style = SnType.headline, color = sn.ink)
        Spacer(Modifier.height(6.dp))
        Text(stringResource(R.string.hist_sub), style = SnType.body, color = sn.mut)
        Spacer(Modifier.height(16.dp))
        Row(horizontalArrangement = Arrangement.spacedBy(7.dp)) {
            days.forEachIndexed { i, c ->
                val has = items.any { sameDay(it.createdAt, c) }; val sel = day == i
                Column(Modifier.weight(1f).height(62.dp).clip(RoundedCornerShape(18.dp)).background(if (sel) sn.surf2 else sn.surf)
                    .border(if (sel) 1.5.dp else 1.dp, if (sel) sn.acc else sn.line2, RoundedCornerShape(18.dp)).clickable(role = Role.Tab) { day = if (sel) null else i },
                    horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.Center) {
                    Text(c.getDisplayName(Calendar.DAY_OF_WEEK, Calendar.SHORT, java.util.Locale.getDefault()) ?: "", style = SnType.caption, color = sn.mut)
                    Text("${c.get(Calendar.DAY_OF_MONTH)}", style = SnType.title, color = sn.ink)
                    Box(Modifier.size(5.dp).clip(RoundedCornerShape(3.dp)).background(if (has) sn.accT else Color.Transparent))
                }
            }
        }
        Spacer(Modifier.height(16.dp))
        if (!history) Text(stringResource(R.string.hist_off), style = SnType.body, color = sn.mut)
        val shown = items.filter { e -> day?.let { sameDay(e.createdAt, days[it]) } ?: true }
        if (history && shown.isEmpty()) {
            Box(Modifier.fillMaxWidth().clip(RoundedCornerShape(22.dp)).border(1.dp, sn.line, RoundedCornerShape(22.dp)).padding(26.dp), contentAlignment = Alignment.Center) {
                Text(stringResource(if (day != null) R.string.hist_empty else R.string.hist_none), style = SnType.body, color = sn.mut)
            }
        }
        shown.forEach { e ->
            HistoryRow(vm, e) { onOpen(e) }
            Spacer(Modifier.height(10.dp))
        }
        if (spots.isNotEmpty()) {
            Overline(stringResource(R.string.hist_spots))
            spots.forEach { s -> SpotRow(s) { onSpot(s.id) }; Spacer(Modifier.height(8.dp)) }
        }
        Spacer(Modifier.height(14.dp))
        OutlineButton("+  " + stringResource(R.string.hist_add), Modifier.fillMaxWidth(), onScan)
        Spacer(Modifier.height(16.dp))
        Box(Modifier.fillMaxWidth().height(140.dp).clip(RoundedCornerShape(24.dp))) {
            PixelImage(R.drawable.px_journey, Modifier.fillMaxSize())
            Text(stringResource(R.string.hist_quote), style = SnType.caption, color = Color(0xFFECE4F2), modifier = Modifier.align(Alignment.Center))
        }
    }
}

@Composable
private fun HistoryRow(vm: SessionViewModel, e: AnalysisEntity, onClick: () -> Unit) {
    val sn = LocalSn.current
    var thumb by remember(e.id) { mutableStateOf<android.graphics.Bitmap?>(null) }
    androidx.compose.runtime.LaunchedEffect(e.id) { thumb = vm.c.images.load(e.imageRef) }
    SnCard(onClick = onClick) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            thumb?.let { Image(it.asImageBitmap(), null, Modifier.size(56.dp).clip(RoundedCornerShape(16.dp)), contentScale = ContentScale.Crop) }
                ?: Box(Modifier.size(56.dp).clip(RoundedCornerShape(16.dp)).background(sn.surf2))
            Spacer(Modifier.width(14.dp))
            Column(Modifier.weight(1f)) {
                Text(DateFormat.getDateTimeInstance(DateFormat.MEDIUM, DateFormat.SHORT).format(Date(e.createdAt)), style = SnType.label, color = sn.ink)
                Text(stringResource(R.string.hist_possible, stringResource(categoryNameRes(e.topKey))), style = SnType.caption, color = sn.mut)
                Text(stringResource(tierRes(e.tier)), style = SnType.caption, color = sn.tier(com.skinnova.app.model.Tier.parse(e.tier) ?: com.skinnova.app.model.Tier.LOW))
            }
            Text("›", color = sn.mut, fontSize = 18.sp)
        }
    }
}

/** 10 Profile: appearance, language, on-device model, privacy & data, about. */
@Composable
fun ProfileScreen(vm: SessionViewModel, onReplayIntro: () -> Unit, onSetup: () -> Unit) {
    val sn = LocalSn.current
    val c = vm.c
    val scope = rememberCoroutineScope()
    val dark by c.settings.dark.collectAsState()
    val lang by c.settings.lang.collectAsState()
    val history by c.settings.history.collectAsState()
    val tts by c.settings.tts.collectAsState()
    val coin by c.settings.coinMm.collectAsState()
    val count by c.db.dao().analysisCount().collectAsState(0)
    val spots by c.db.dao().spots().collectAsState(emptyList())
    var confirm by remember { mutableStateOf(false) }
    var alsoModel by remember { mutableStateOf(false) }
    val model = remember { c.models.activeModel() }
    val cvInfo: CvPreprocess? = remember { c.cv.pre }
    Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).statusBarsPadding().padding(start = 18.dp, end = 18.dp, top = 18.dp, bottom = 110.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Box(Modifier.size(60.dp).clip(RoundedCornerShape(22.dp)).background(sn.surf2).border(1.5.dp, sn.acc, RoundedCornerShape(22.dp)), contentAlignment = Alignment.Center) {
                Text("G", fontFamily = com.skinnova.app.ui.theme.Pixelify, fontSize = 26.sp, color = sn.accT)
            }
            Spacer(Modifier.width(14.dp))
            Column {
                Text(stringResource(R.string.prof_guest), style = SnType.titleL, color = sn.ink)
                Text(stringResource(R.string.prof_summary, count, spots.size), style = SnType.caption, color = sn.mut)
            }
        }
        Overline(stringResource(R.string.prof_appearance))
        Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            Chip(stringResource(R.string.prof_dark), dark, { c.settings.setDark(true) }, Modifier.weight(1f))
            Chip(stringResource(R.string.prof_light), !dark, { c.settings.setDark(false) }, Modifier.weight(1f))
        }
        Overline(stringResource(R.string.prof_language))
        Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            Chip("English", lang == "en", { c.settings.setLang("en") }, Modifier.weight(1f))
            Chip("हिन्दी", lang == "hi", { c.settings.setLang("hi") }, Modifier.weight(1f))
        }
        Overline(stringResource(R.string.prof_model))
        SnCard(onClick = onSetup) {
            Column {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(stringResource(R.string.prof_model_name), style = SnType.title, color = sn.ink, modifier = Modifier.weight(1f))
                    com.skinnova.app.ui.components.SmallTag(if (model?.id?.startsWith("skinnova") == true) stringResource(R.string.prof_model_badge)
                        else model?.let { "stock" } ?: stringResource(R.string.prof_model_missing), sn.accT)
                }
                Spacer(Modifier.height(12.dp))
                KV(stringResource(R.string.prof_runtime), "LiteRT-LM")
                KV(stringResource(R.string.prof_vision), cvInfo?.let { "SkinNova CV · ${it.classes.size} classes" } ?: "—")
                KV(stringResource(R.string.prof_backend), c.engineHolder.backendName)
                KV(stringResource(R.string.prof_network), stringResource(R.string.prof_network_none))
            }
        }
        Overline(stringResource(R.string.prof_privacy))
        SnCard(framed = false) {
            Column {
                ToggleRow(stringResource(R.string.prof_history), stringResource(R.string.prof_history_sub), history) { c.settings.setHistory(!history) }
                ToggleRow(stringResource(R.string.prof_tts), stringResource(R.string.prof_tts_sub), tts) { c.settings.setTts(!tts) }
                Text(stringResource(R.string.prof_coin), style = SnType.bodyL, color = sn.ink, modifier = Modifier.padding(top = 12.dp))
                Text(stringResource(R.string.prof_coin_sub, coin), style = SnType.micro, color = sn.mut)
                Row(horizontalArrangement = Arrangement.spacedBy(6.dp), modifier = Modifier.padding(vertical = 8.dp)) {
                    listOf(20.0, 21.93, 23.0, 25.0, 27.0).forEach { mm -> Chip("%.1f".format(mm), coin == mm, { c.settings.setCoinMm(mm) }) }
                }
                Row(Modifier.fillMaxWidth().heightIn(min = 56.dp).clickable(role = Role.Button, onClick = onReplayIntro), verticalAlignment = Alignment.CenterVertically) {
                    Text(stringResource(R.string.prof_replay), style = SnType.bodyL, color = sn.ink, modifier = Modifier.weight(1f)); Text("›", color = sn.mut)
                }
                Row(Modifier.fillMaxWidth().heightIn(min = 56.dp).clickable(role = Role.Button) { confirm = true }, verticalAlignment = Alignment.CenterVertically) {
                    Text(stringResource(R.string.prof_clear), style = SnType.bodyL, color = Color(0xFFE0787A), modifier = Modifier.weight(1f))
                    Text("$count", style = SnType.caption, color = sn.mut)
                }
            }
        }
        Overline(stringResource(R.string.prof_about))
        SnCard(framed = false) {
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text(stringResource(R.string.about_sources), style = SnType.label, color = sn.ink)
                Text(stringResource(R.string.about_sources_body), style = SnType.caption, color = sn.mut)
                Text(stringResource(R.string.about_limits), style = SnType.label, color = sn.ink)
                Text(stringResource(R.string.about_limits_body), style = SnType.caption, color = sn.mut)
                Text(stringResource(R.string.about_version, BuildConfig.VERSION_NAME, c.prompts.version, model?.sha256?.take(12) ?: "—"), style = SnType.micro, color = sn.mut)
            }
        }
        Spacer(Modifier.height(16.dp))
        Box(Modifier.fillMaxWidth().clip(RoundedCornerShape(20.dp)).background(sn.surf2).padding(16.dp)) {
            Text(stringResource(R.string.prof_about_body), style = SnType.caption, color = sn.mut)
        }
    }
    if (confirm) AlertDialog(onDismissRequest = { confirm = false },
        confirmButton = { TextButton({ confirm = false; scope.launch { deleteEverything(vm, alsoModel) } }) { Text(stringResource(R.string.delete), color = Color(0xFFE0787A)) } },
        dismissButton = { TextButton({ confirm = false }) { Text(stringResource(R.string.cancel)) } },
        text = {
            Column {
                Text(stringResource(R.string.prof_clear_confirm))
                Row(Modifier.padding(top = 12.dp).clickable(role = Role.Checkbox) { alsoModel = !alsoModel }, verticalAlignment = Alignment.CenterVertically) {
                    Toggle(alsoModel); Spacer(Modifier.width(10.dp)); Text(stringResource(R.string.prof_clear_model))
                }
            }
        })
}

/** "Delete everything" (architecture §9): DB, encrypted images, temp files; model only if selected. */
suspend fun deleteEverything(vm: SessionViewModel, alsoModel: Boolean) {
    val c = vm.c
    kotlinx.coroutines.withContext(kotlinx.coroutines.Dispatchers.IO) {
        val d = c.db.dao(); d.clearMetrics(); d.clearCaptures(); d.clearSpots(); d.clearAnalyses()
        c.images.deleteAll()
        c.ctx.cacheDir.listFiles()?.forEach { it.deleteRecursively() }
        androidx.work.WorkManager.getInstance(c.ctx).cancelAllWorkByTag("spot_reminder")
        if (alsoModel) { c.engineHolder.release(); c.models.deleteModels() }
    }
}

@Composable
private fun KV(k: String, v: String) {
    val sn = LocalSn.current
    Row(Modifier.fillMaxWidth().padding(vertical = 4.dp)) { Text(k, style = SnType.caption, color = sn.mut, modifier = Modifier.weight(1f)); Text(v, style = SnType.caption, color = sn.ink) }
}

@Composable
private fun ToggleRow(label: String, sub: String, on: Boolean, onClick: () -> Unit) {
    val sn = LocalSn.current
    Row(Modifier.fillMaxWidth().heightIn(min = 62.dp).clickable(role = Role.Switch, onClick = onClick), verticalAlignment = Alignment.CenterVertically) {
        Column(Modifier.weight(1f)) { Text(label, style = SnType.bodyL, color = sn.ink); Text(sub, style = SnType.micro, color = sn.mut) }
        Toggle(on)
    }
}
