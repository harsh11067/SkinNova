package com.skinnova.app.ui.screens

import android.Manifest
import android.content.pm.PackageManager
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
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
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import com.skinnova.app.R
import com.skinnova.app.ui.Draft
import com.skinnova.app.ui.SessionViewModel
import com.skinnova.app.ui.VoiceState
import com.skinnova.app.ui.components.Bar
import com.skinnova.app.ui.components.Chip
import com.skinnova.app.ui.components.OutlineButton
import com.skinnova.app.ui.components.PrimaryButton
import com.skinnova.app.ui.components.SnCard
import com.skinnova.app.ui.theme.LocalSn
import com.skinnova.app.ui.theme.SnType

private data class Opt(val value: Any, val label: Int, val hint: Int? = null)
private enum class Kind { CHOICE, SCALE, YESNO, TEXT }
private data class Step(val key: String, val title: Int, val sub: Int, val kind: Kind, val opts: List<Opt> = emptyList(), val optional: Boolean = false)

private val SITES = listOf("face" to R.string.site_face, "scalp" to R.string.site_scalp, "neck" to R.string.site_neck, "chest" to R.string.site_chest,
    "back" to R.string.site_back, "abdomen" to R.string.site_abdomen, "arm" to R.string.site_arm, "hand" to R.string.site_hand,
    "leg" to R.string.site_leg, "foot" to R.string.site_foot, "groin" to R.string.site_groin, "nails" to R.string.site_nails, "other" to R.string.site_other)
private val LEVELS = listOf(Opt(0, R.string.lvl_0, R.string.lvl_0_h), Opt(1, R.string.lvl_1, R.string.lvl_1_h), Opt(2, R.string.lvl_2, R.string.lvl_2_h), Opt(3, R.string.lvl_3, R.string.lvl_3_h))

private val STEPS = listOf(
    Step("body_site", R.string.q_site, R.string.q_site_sub, Kind.CHOICE, SITES.map { Opt(it.first, it.second) }),
    Step("duration", R.string.q_duration, R.string.q_duration_sub, Kind.CHOICE, listOf(Opt("lt_1w", R.string.dur_lt_1w), Opt("1_4w", R.string.dur_1_4w), Opt("1_6m", R.string.dur_1_6m), Opt("gt_6m", R.string.dur_gt_6m))),
    Step("itch", R.string.q_itch, R.string.q_scale_sub, Kind.SCALE, LEVELS),
    Step("pain", R.string.q_pain, R.string.q_scale_sub, Kind.SCALE, LEVELS),
    Step("changing", R.string.q_changing, R.string.q_changing_sub, Kind.CHOICE, listOf(Opt("no", R.string.chg_no), Opt("growing", R.string.chg_growing),
        Opt("changing_color", R.string.chg_changing_color), Opt("changing_shape", R.string.chg_changing_shape), Opt("spreading", R.string.chg_spreading), Opt("unsure", R.string.chg_unsure))),
    Step("bleeding_or_crusting", R.string.q_bleeding, R.string.q_yesno_sub, Kind.YESNO),
    Step("fever_or_unwell", R.string.q_fever, R.string.q_yesno_sub, Kind.YESNO),
    Step("others_affected", R.string.q_others, R.string.q_yesno_sub, Kind.YESNO),
    Step("new_product_or_exposure", R.string.q_product, R.string.q_yesno_sub, Kind.YESNO),
    Step("age_band", R.string.q_age, R.string.q_duration_sub, Kind.CHOICE, listOf(Opt("lt_12", R.string.age_lt_12), Opt("12_17", R.string.age_12_17), Opt("18_39", R.string.age_18_39), Opt("40_59", R.string.age_40_59), Opt("60_plus", R.string.age_60_plus))),
    Step("skin_tone", R.string.q_tone, R.string.q_tone_sub, Kind.CHOICE, listOf(Opt("fitz_1_2", R.string.tone_fitz_1_2), Opt("fitz_3_4", R.string.tone_fitz_3_4), Opt("fitz_5_6", R.string.tone_fitz_5_6), Opt("unknown", R.string.tone_unknown)), optional = true),
    Step("free_text", R.string.q_notes, R.string.q_notes_sub, Kind.TEXT, optional = true),
)

private fun Draft.get(k: String): Any? = when (k) {
    "body_site" -> bodySite; "duration" -> duration; "itch" -> itch; "pain" -> pain; "changing" -> changing
    "bleeding_or_crusting" -> bleeding; "fever_or_unwell" -> fever; "others_affected" -> others; "new_product_or_exposure" -> product
    "age_band" -> ageBand; "skin_tone" -> skinTone; "free_text" -> freeText; else -> null
}

private fun Draft.set(k: String, v: Any?): Draft = when (k) {
    "body_site" -> copy(bodySite = v as String); "duration" -> copy(duration = v as String); "itch" -> copy(itch = v as Int); "pain" -> copy(pain = v as Int)
    "changing" -> copy(changing = v as String); "bleeding_or_crusting" -> copy(bleeding = v as Boolean); "fever_or_unwell" -> copy(fever = v as Boolean)
    "others_affected" -> copy(others = v as Boolean); "new_product_or_exposure" -> copy(product = v as Boolean); "age_band" -> copy(ageBand = v as String)
    "skin_tone" -> copy(skinTone = v as String); "free_text" -> copy(freeText = (v as String).take(200)); else -> this
}.let { d -> if (k in evidence) d.copy(evidence = evidence - k) else d }   // a manual change replaces the voice answer

@OptIn(ExperimentalLayoutApi::class)
@Composable
fun QuestionsScreen(vm: SessionViewModel, onBack: () -> Unit, onAnalyze: () -> Unit) {
    val sn = LocalSn.current
    val draft by vm.draft.collectAsState()
    val photo by vm.photo.collectAsState()
    var i by remember { mutableIntStateOf(0) }
    var voiceOpen by remember { mutableStateOf(false) }
    LaunchedEffect(Unit) { vm.warmUp() }
    val step = STEPS[i]
    val value = draft.get(step.key)
    val answered = step.optional || value != null
    Column(Modifier.fillMaxSize().statusBarsPadding().navigationBarsPadding().imePadding().padding(horizontal = 18.dp, vertical = 8.dp)) {
        Row(Modifier.fillMaxWidth().height(48.dp), verticalAlignment = Alignment.CenterVertically) {
            Box(Modifier.size(48.dp).clickable(role = Role.Button) { if (i == 0) onBack() else i-- }.semantics { contentDescription = "Back" },
                contentAlignment = Alignment.Center) { Text("←", color = sn.ink, fontSize = 20.sp) }
            Text(stringResource(R.string.q_step, i + 1, STEPS.size), style = SnType.body, color = sn.mut, modifier = Modifier.weight(1f))
            photo?.let { Image(it.asImageBitmap(), null, Modifier.size(40.dp).clip(RoundedCornerShape(13.dp)).border(1.5.dp, sn.acc, RoundedCornerShape(13.dp)), contentScale = ContentScale.Crop) }
        }
        Spacer(Modifier.height(10.dp))
        Row(horizontalArrangement = Arrangement.spacedBy(4.dp)) {
            STEPS.forEachIndexed { k, s ->
                val done = s.optional || draft.get(s.key) != null
                Box(Modifier.weight(1f).height(6.dp).clip(RoundedCornerShape(3.dp)).background(when { k == i -> sn.acc; done -> sn.pur; else -> sn.track }))
            }
        }
        Spacer(Modifier.height(22.dp))
        Text(stringResource(step.title), style = SnType.display, color = sn.ink)
        Spacer(Modifier.height(8.dp))
        Text(stringResource(step.sub), style = SnType.body, color = sn.mut)
        draft.evidence[step.key]?.let { ev ->
            Spacer(Modifier.height(10.dp))
            Box(Modifier.clip(RoundedCornerShape(12.dp)).background(sn.surf2).padding(horizontal = 12.dp, vertical = 8.dp)) {
                Text("🎙 " + stringResource(R.string.v_evidence, ev), style = SnType.caption, color = sn.accT)
            }
        }
        if (!answered && draft.viaVoice) { Spacer(Modifier.height(6.dp)); Text(stringResource(R.string.q_unanswered), style = SnType.caption, color = sn.moderate) }
        Spacer(Modifier.height(18.dp))
        Column(Modifier.weight(1f).verticalScroll(rememberScrollState())) {
            when (step.kind) {
                Kind.CHOICE -> FlowRow(horizontalArrangement = Arrangement.spacedBy(10.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                    step.opts.forEach { o -> Chip(stringResource(o.label), value == o.value, { vm.update { it.set(step.key, o.value) } }) }
                }
                Kind.SCALE -> Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                    step.opts.forEach { o ->
                        val sel = value == o.value
                        Row(Modifier.fillMaxWidth().heightIn(min = 58.dp).clip(RoundedCornerShape(20.dp)).background(if (sel) sn.surf2 else sn.surf)
                            .border(if (sel) 1.5.dp else 1.dp, if (sel) sn.acc else sn.line2, RoundedCornerShape(20.dp))
                            .clickable(role = Role.RadioButton) { vm.update { it.set(step.key, o.value) } }.padding(horizontal = 18.dp),
                            verticalAlignment = Alignment.CenterVertically) {
                            Column(Modifier.weight(1f)) {
                                Text(stringResource(o.label), style = SnType.label, color = if (sel) sn.accT else sn.ink)
                                o.hint?.let { Text(stringResource(it), style = SnType.micro, color = sn.mut) }
                            }
                            Row(horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                                repeat(3) { b -> Box(Modifier.size(12.dp).clip(RoundedCornerShape(3.dp)).background(if (b < (o.value as Int)) sn.acc else sn.track)) }
                            }
                        }
                    }
                }
                Kind.YESNO -> Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                    Chip(stringResource(R.string.yes), value == true, { vm.update { it.set(step.key, true) } }, Modifier.weight(1f))
                    Chip(stringResource(R.string.no), value == false, { vm.update { it.set(step.key, false) } }, Modifier.weight(1f))
                }
                Kind.TEXT -> {
                    BasicTextField(draft.freeText, { t -> vm.update { it.set("free_text", t) } },
                        Modifier.fillMaxWidth().height(150.dp).clip(RoundedCornerShape(22.dp)).background(sn.surf).border(1.dp, sn.line, RoundedCornerShape(22.dp)).padding(16.dp),
                        textStyle = SnType.bodyL.copy(color = sn.ink), cursorBrush = SolidColor(sn.acc),
                        decorationBox = { inner -> if (draft.freeText.isEmpty()) Text(stringResource(R.string.q_notes_hint), style = SnType.bodyL, color = sn.mut); inner() })
                    Spacer(Modifier.height(14.dp))
                    Box(Modifier.fillMaxWidth().clip(RoundedCornerShape(20.dp)).background(sn.surf2).padding(14.dp)) {
                        Text("⌂  " + stringResource(R.string.q_private), style = SnType.caption, color = sn.mut)
                    }
                }
            }
        }
        if (i == 0 || draft.viaVoice.not()) {
            Box(Modifier.fillMaxWidth().heightIn(min = 48.dp).clip(RoundedCornerShape(24.dp)).border(1.dp, sn.acc, RoundedCornerShape(24.dp))
                .clickable(role = Role.Button) { voiceOpen = true }, contentAlignment = Alignment.Center) {
                Text("🎙  " + stringResource(R.string.q_speak), style = SnType.bodyL, color = sn.accT)
            }
            Spacer(Modifier.height(10.dp))
        }
        val last = i == STEPS.lastIndex
        PrimaryButton(stringResource(if (last) R.string.q_analyze else if (step.optional && value == null) R.string.q_skip else R.string.q_next),
            enabled = if (last) draft.complete() else answered) {
            if (last) onAnalyze() else i++
        }
    }
    if (voiceOpen) VoiceSheet(vm) { voiceOpen = false; vm.resetVoice()
        // jump to the first question the voice intake did not answer
        val d = vm.draft.value; val k = STEPS.indexOfFirst { !it.optional && d.get(it.key) == null }; if (k >= 0) i = k
    }
}

/** USP-2: record → "Here's what we heard" (editable) → extract → validated pre-fill. */
@Composable
fun VoiceSheet(vm: SessionViewModel, onClose: () -> Unit) {
    val sn = LocalSn.current
    val ctx = LocalContext.current
    val st by vm.voice.collectAsState()
    var transcript by remember { mutableStateOf("") }
    var denied by remember { mutableStateOf(false) }
    val perm = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { ok -> if (ok) vm.startRecording() else denied = true }
    LaunchedEffect(st) { (st as? VoiceState.Heard)?.let { transcript = it.transcript } }
    Box(Modifier.fillMaxSize().background(sn.bg.copy(alpha = .94f)).clickable(enabled = false) {}) {
        Column(Modifier.align(Alignment.BottomCenter).fillMaxWidth().navigationBarsPadding().imePadding().padding(18.dp)) {
            SnCard {
                Column {
                    Text(stringResource(R.string.v_title), style = SnType.titleL, color = sn.ink)
                    Spacer(Modifier.height(6.dp))
                    Text(stringResource(R.string.v_sub), style = SnType.body, color = sn.mut)
                    Spacer(Modifier.height(16.dp))
                    when (val s = st) {
                        is VoiceState.Idle -> {
                            if (denied) Text(stringResource(R.string.v_mic_denied), style = SnType.body, color = sn.moderate)
                            else Box(Modifier.align(Alignment.CenterHorizontally).size(84.dp).clip(CircleShape).background(sn.acc)
                                .clickable(role = Role.Button) {
                                    if (ContextCompat.checkSelfPermission(ctx, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) vm.startRecording()
                                    else perm.launch(Manifest.permission.RECORD_AUDIO)
                                }.semantics { contentDescription = "Start recording" }, contentAlignment = Alignment.Center) { Text("🎙", fontSize = 30.sp) }
                        }
                        is VoiceState.Recording -> {
                            Text(stringResource(R.string.v_listening) + "  ${s.ms / 1000}s / 30s", style = SnType.label, color = sn.ink)
                            Spacer(Modifier.height(10.dp)); Bar(s.level, sn.acc); Spacer(Modifier.height(14.dp))
                            OutlineButton(stringResource(R.string.v_stop), Modifier.fillMaxWidth()) { vm.stopRecording() }
                        }
                        is VoiceState.Transcribing -> Text(stringResource(R.string.v_transcribing), style = SnType.label, color = sn.ink)
                        is VoiceState.Heard -> {
                            Text(stringResource(R.string.v_heard), style = SnType.label, color = sn.ink)
                            Text(stringResource(R.string.v_heard_sub), style = SnType.caption, color = sn.mut)
                            Spacer(Modifier.height(8.dp))
                            BasicTextField(transcript, { transcript = it }, Modifier.fillMaxWidth().heightIn(min = 90.dp).clip(RoundedCornerShape(16.dp))
                                .background(sn.surf2).padding(12.dp), textStyle = SnType.bodyL.copy(color = sn.ink), cursorBrush = SolidColor(sn.acc))
                            Spacer(Modifier.height(12.dp))
                            PrimaryButton(stringResource(R.string.v_extract), enabled = transcript.isNotBlank()) { vm.extract(transcript) }
                        }
                        is VoiceState.Extracting -> Text(stringResource(R.string.v_extracting), style = SnType.label, color = sn.ink)
                        is VoiceState.Filled -> {
                            Text(if (s.n > 0) stringResource(R.string.v_filled, s.n) else stringResource(R.string.v_none), style = SnType.label, color = sn.ink)
                            Spacer(Modifier.height(12.dp)); PrimaryButton(stringResource(R.string.v_confirm)) { onClose() }
                        }
                        is VoiceState.Error -> {
                            Text(stringResource(when (s.kind) { "model" -> R.string.v_needs_model; "speech_pack" -> R.string.v_needs_speech_pack; else -> R.string.v_unclear }),
                                style = SnType.body, color = sn.moderate)
                            Spacer(Modifier.height(12.dp))
                            if (s.kind == "unclear") OutlineButton(stringResource(R.string.v_retry), Modifier.fillMaxWidth()) { vm.resetVoice() }
                        }
                    }
                    Spacer(Modifier.height(10.dp))
                    Text(stringResource(R.string.cancel), style = SnType.bodyL, color = sn.mut,
                        modifier = Modifier.align(Alignment.CenterHorizontally).clickable(role = Role.Button) { vm.stopRecording(); onClose() }.padding(12.dp))
                }
            }
        }
    }
}
