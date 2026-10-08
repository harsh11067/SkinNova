package com.skinnova.app.ui.screens

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
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.skinnova.app.R
import com.skinnova.app.ml.AnalysisState
import com.skinnova.app.model.FinalResult
import com.skinnova.app.model.Mode
import com.skinnova.app.model.Tier
import com.skinnova.app.ui.SessionViewModel
import com.skinnova.app.ui.components.Bar
import com.skinnova.app.ui.components.Disclaimer
import com.skinnova.app.ui.components.OutlineButton
import com.skinnova.app.ui.components.SpeakButton
import com.skinnova.app.ui.components.RedFlagBanner
import com.skinnova.app.ui.components.SmallTag
import com.skinnova.app.ui.components.SnCard
import com.skinnova.app.ui.components.SpinRing
import com.skinnova.app.ui.components.TopBar
import com.skinnova.app.ui.components.TriageCard
import com.skinnova.app.ui.components.categoryNameRes
import com.skinnova.app.ui.components.ruleMessageRes
import com.skinnova.app.ui.theme.LocalSn
import com.skinnova.app.ui.theme.SnType

@Composable
fun AnalyzingScreen(vm: SessionViewModel, onDone: () -> Unit, onCancel: () -> Unit) {
    val sn = LocalSn.current
    val st by vm.state.collectAsState()
    val photo by vm.photo.collectAsState()
    androidx.compose.runtime.LaunchedEffect(st) { if (st is AnalysisState.Done) onDone() }
    val order = listOf(R.string.an_step_photo, R.string.an_step_cv, R.string.an_step_rules, R.string.an_step_llm, R.string.an_step_translate)
    val idx = when (st) {
        AnalysisState.CheckingPhoto -> 0; AnalysisState.Classifying -> 1; AnalysisState.Rules -> 2
        AnalysisState.LoadingModel, is AnalysisState.Generating, AnalysisState.Validating -> 3; AnalysisState.Translating -> 4; else -> 0
    }
    val early0 by vm.early.collectAsState()
    val ring = if (early0 != null) 150.dp else 220.dp   // smaller once the early look needs the space
    Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).statusBarsPadding().navigationBarsPadding().padding(horizontal = 26.dp, vertical = 24.dp),
        horizontalAlignment = Alignment.CenterHorizontally) {
        Box(Modifier.size(ring), contentAlignment = Alignment.Center) {
            val llmBusy = st is AnalysisState.LoadingModel || st is AnalysisState.Generating || st is AnalysisState.Validating
            SpinRing(ring, sn.line, 18000, dotted = false, spinning = !llmBusy)
            Box(Modifier.size(ring * 0.84f), contentAlignment = Alignment.Center) { SpinRing(ring * 0.84f, sn.acc, 9000, reverse = true, spinning = !llmBusy) }
            photo?.let { Image(it.asImageBitmap(), null, Modifier.size(ring * 0.65f).clip(CircleShape), contentScale = ContentScale.Crop) }
        }
        Spacer(Modifier.height(30.dp))
        Text(stringResource(R.string.an_title), style = SnType.headline, color = sn.ink)
        Spacer(Modifier.height(8.dp))
        Text(stringResource(R.string.an_sub), style = SnType.caption, color = sn.mut)
        Spacer(Modifier.height(24.dp))
        SnCard(framed = false) {
            Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                order.forEachIndexed { k, l ->
                    if (k == 4 && vm.c.settings.lang.value == "en") return@forEachIndexed
                    val done = k < idx; val cur = k == idx
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Box(Modifier.size(22.dp).clip(RoundedCornerShape(8.dp)).background(if (done) sn.acc else sn.surf2)
                            .border(1.dp, if (cur) sn.acc else sn.line2, RoundedCornerShape(8.dp)), contentAlignment = Alignment.Center) {
                            Text(if (done) "✓" else if (cur) "•" else "", color = sn.onAcc, fontSize = 11.sp)
                        }
                        Spacer(Modifier.width(12.dp))
                        Text(stringResource(l), style = SnType.body, color = if (done || cur) sn.ink else sn.mut)
                    }
                }
            }
        }
        Spacer(Modifier.height(14.dp))
        val early by vm.early.collectAsState()
        early?.let { EarlyLookCard(it) }
        when (val s = st) {
            AnalysisState.LoadingModel -> Text(stringResource(R.string.an_loading_model), style = SnType.caption, color = sn.mut)
            is AnalysisState.Generating -> if (s.partialText.isEmpty()) Text(stringResource(R.string.an_reading), style = SnType.caption, color = sn.mut)
                else Text(s.partialText.takeLast(240), style = SnType.micro, color = sn.mut, maxLines = 5)
            is AnalysisState.Failed -> Text(s.message, style = SnType.caption, color = sn.urgent)
            else -> {}
        }
        Spacer(Modifier.height(16.dp))
        KeepRunningTip()
        OutlineButton(stringResource(R.string.an_cancel), Modifier.fillMaxWidth()) { vm.cancel(); onCancel() }
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
fun ResultScreen(vm: SessionViewModel, onHome: () -> Unit, onRetake: () -> Unit, onLearn: (String) -> Unit, onTrack: () -> Unit) {
    val sn = LocalSn.current
    val res by vm.result.collectAsState()
    val photo by vm.photo.collectAsState()
    val history by vm.c.settings.history.collectAsState()
    val tts by vm.c.settings.tts.collectAsState()
    val r: FinalResult = res ?: return
    var showHi by remember { mutableStateOf(r.localized != null && r.lang == "hi") }
    var saved by remember(r.createdAt) { mutableStateOf(vm.savedId != null) }
    var askSave by remember { mutableStateOf(false) }
    val loc = if (showHi) r.localized else null
    val lang = if (showHi) "hi" else "en"
    val topOther = r.output.possibleCategories.firstOrNull()?.key == "other"
    val uncertain = topOther || r.output.uncertainty.level == "high" || (r.cvTop3.firstOrNull()?.p ?: 0.0) < 0.5
    val appCtx = androidx.compose.ui.platform.LocalContext.current.applicationContext
    androidx.compose.runtime.LaunchedEffect(r.createdAt) { com.skinnova.app.notify.Notifier.cancelReady(appCtx) }   // seen → clear "ready"
    val tier = Tier.parse(r.finalTier) ?: Tier.MODERATE
    val p = r.cvTop3.associate { it.key to it.p }
    Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).statusBarsPadding().navigationBarsPadding().padding(horizontal = 18.dp, vertical = 8.dp)) {
        TopBar(stringResource(R.string.res_title), onHome)
        Row(verticalAlignment = Alignment.CenterVertically) {
            photo?.let {
                Box(Modifier.size(112.dp).clip(RoundedCornerShape(26.dp)).border(1.5.dp, sn.acc, RoundedCornerShape(26.dp)).padding(4.dp)) {
                    Image(it.asImageBitmap(), null, Modifier.fillMaxSize().clip(RoundedCornerShape(21.dp)), contentScale = ContentScale.Crop)
                }
            }
            Spacer(Modifier.width(14.dp))
            FlowRow(Modifier.weight(1f), horizontalArrangement = Arrangement.spacedBy(6.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                Text(stringResource(R.string.res_answers), style = SnType.overline.copy(fontSize = 10.sp), color = sn.mut, modifier = Modifier.fillMaxWidth())
                answerChips(r).forEach { SmallTag(it) }
            }
        }
        Spacer(Modifier.height(14.dp))
        // 1 red flags (strings.xml, never LLM text)
        val msgs = r.ruleMessages.mapNotNull { k -> ruleMessageRes(k)?.let { if (k.startsWith("rf_t")) stringResource(it, "") else stringResource(it) } }
        if (msgs.isNotEmpty()) { RedFlagBanner(msgs); Spacer(Modifier.height(12.dp)) }
        // 2 triage
        TriageCard(tier)
        Spacer(Modifier.height(12.dp))
        if (r.mode == Mode.basic) {
            Box(Modifier.fillMaxWidth().clip(RoundedCornerShape(20.dp)).background(sn.surf2).padding(14.dp)) {
                Text(stringResource(R.string.res_basic_mode), style = SnType.caption, color = sn.mut)
            }
            Spacer(Modifier.height(12.dp))
        }
        // 2b no clear match: "other" leads → say what that means and show the closest known conditions (never a dead end)
        if (topOther) {
            Box(Modifier.fillMaxWidth().clip(RoundedCornerShape(20.dp)).background(sn.surf2).padding(14.dp)) {
                Column {
                    Text(stringResource(R.string.res_nomatch_title), style = SnType.label, color = sn.ink)
                    Spacer(Modifier.height(4.dp))
                    Text(stringResource(R.string.res_nomatch_body), style = SnType.caption, color = sn.mut)
                }
            }
            Spacer(Modifier.height(12.dp))
        }
        // 3 possible categories (likelihood words; bar = calibrated image-model score, no big numbers)
        SnCard {
            Column {
                Text(stringResource(R.string.res_possible), style = SnType.title, color = sn.ink)
                Spacer(Modifier.height(4.dp))
                Text(stringResource(R.string.res_possible_sub), style = SnType.caption, color = sn.mut)
                Spacer(Modifier.height(14.dp))
                // "other" first and fewer than 3 rows → add the image model's next closest known conditions (as less likely)
                val shown = r.output.possibleCategories.map { it.key }.toSet()
                val extra = if (!topOther) emptyList() else r.cvTop3.filter { it.key !in shown && it.key != "other" }
                    .take(3 - r.output.possibleCategories.size.coerceAtMost(3))
                    .map { com.skinnova.app.model.Category(it.key, "less_likely", stringResource(R.string.res_nomatch_why)) }
                (r.output.possibleCategories + extra).forEachIndexed { k, c ->
                    Column(Modifier.fillMaxWidth().clip(RoundedCornerShape(14.dp)).clickable(role = Role.Button) { onLearn(c.key) }.padding(vertical = 8.dp, horizontal = 4.dp)) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Box(Modifier.size(22.dp).clip(CircleShape).border(1.dp, sn.line, CircleShape), contentAlignment = Alignment.Center) {
                                Text("${k + 1}", fontSize = 10.sp, color = sn.mut)
                            }
                            Spacer(Modifier.width(10.dp))
                            Text(stringResource(categoryNameRes(c.key)), style = SnType.label, color = sn.ink, modifier = Modifier.weight(1f))
                            LikelihoodPill(c.likelihood)
                        }
                        Spacer(Modifier.height(7.dp))
                        Bar(((p[c.key] ?: 0.0)).toFloat().coerceAtLeast(0.03f), modifier = Modifier.padding(start = 32.dp))
                        Spacer(Modifier.height(6.dp))
                        Text(loc?.categoryWhy?.get(c.key) ?: c.why, style = SnType.caption, color = sn.mut, modifier = Modifier.padding(start = 32.dp))
                    }
                }
            }
        }
        Spacer(Modifier.height(12.dp))
        // 4 uncertainty
        val lvl = r.output.uncertainty.level
        SnCard(framed = false) {
            Column {
                Row { Text(stringResource(R.string.res_uncertainty), style = SnType.title, color = sn.ink, modifier = Modifier.weight(1f))
                    Text(stringResource(when (lvl) { "low" -> R.string.res_unc_low; "moderate" -> R.string.res_unc_moderate; else -> R.string.res_unc_high }), style = SnType.label, color = sn.accT) }
                Spacer(Modifier.height(10.dp))
                Bar(when (lvl) { "low" -> .85f; "moderate" -> .5f; else -> .2f })
                Spacer(Modifier.height(10.dp))
                r.output.uncertainty.reasons.forEach { Text("• $it", style = SnType.caption, color = sn.mut) }
                if (r.output.disagreementWithImageModel) Text("• " + stringResource(R.string.res_disagree), style = SnType.caption, color = sn.mut)
            }
        }
        Spacer(Modifier.height(12.dp))
        // 5 explanation (+ language toggle, read aloud)
        SnCard(framed = false) {
            Column {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(stringResource(R.string.res_explanation), style = SnType.title, color = sn.ink, modifier = Modifier.weight(1f))
                    if (r.localized != null) {
                        SmallTag(if (showHi) stringResource(R.string.res_lang_en) else stringResource(R.string.res_lang_hi))
                        Spacer(Modifier.width(4.dp))
                        val switchLabel = stringResource(if (showHi) R.string.res_lang_en else R.string.res_lang_hi)
                        Box(Modifier.size(48.dp).semantics { contentDescription = switchLabel }.clickable(role = Role.Button) { showHi = !showHi },
                            contentAlignment = Alignment.Center) { Text("⇄", color = sn.accT) }
                    }
                    if (tts) SpeakButton({ loc?.explanation ?: r.output.explanation }, lang)
                }
                // on-demand Hindi from the fine-tuned model (only full results: Basic mode has no LLM text to translate)
                if (r.localized == null && r.mode == Mode.full && vm.c.llm.available) {
                    val busy by vm.translating.collectAsState()
                    val failMsg = stringResource(R.string.res_translate_fail)
                    val ctx2 = androidx.compose.ui.platform.LocalContext.current
                    Box(Modifier.heightIn(min = 48.dp).clip(RoundedCornerShape(14.dp)).clickable(enabled = !busy, role = Role.Button) {
                        vm.translateResult { android.widget.Toast.makeText(ctx2, failMsg, android.widget.Toast.LENGTH_LONG).show() }
                    }.padding(horizontal = 4.dp), contentAlignment = Alignment.CenterStart) {
                        Text(if (busy) stringResource(R.string.res_translating) else "अ  " + stringResource(R.string.res_translate),
                            style = SnType.label, color = sn.accT)
                    }
                }
                androidx.compose.runtime.LaunchedEffect(r.localized) { if (r.localized != null && !showHi) showHi = true }
                Spacer(Modifier.height(8.dp))
                Text(loc?.explanation ?: r.output.explanation, style = SnType.bodyL, color = sn.ink)
            }
        }
        Spacer(Modifier.height(12.dp))
        BulletCard(stringResource(R.string.res_help), loc?.whatWouldHelp ?: r.output.whatWouldHelp, tts, lang)
        Spacer(Modifier.height(12.dp))
        BulletCard(stringResource(R.string.res_care), loc?.selfCareInfo ?: r.output.selfCareInfo, tts, lang)
        Spacer(Modifier.height(12.dp))
        // 6b home care, food and pharmacy relief: curated + cited (assets/care/relief.json), never LLM text
        ReliefCard(vm, r, tier, tts, lang)
        Spacer(Modifier.height(12.dp))
        // 6c Ask SkinNova: follow-up questions, answered on the phone by Gemma (guarded)
        AskCard(vm, tts, lang)
        Spacer(Modifier.height(12.dp))
        Disclaimer()
        Spacer(Modifier.height(16.dp))
        if (uncertain) {   // the honest way to a surer answer: a better photo (a 2nd photo of the same spot added < 2 pts top-3 on val)
            Text(stringResource(R.string.res_retake_tip), style = SnType.caption, color = sn.mut)
            Spacer(Modifier.height(10.dp))
        }
        Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
            OutlineButton(stringResource(R.string.res_learn), Modifier.weight(1f)) { onLearn(r.output.possibleCategories.first().key) }
            OutlineButton(stringResource(if (saved) R.string.res_saved else R.string.res_save), Modifier.weight(1f)) {
                when {
                    saved -> {}
                    history -> { vm.save(); saved = true }
                    else -> askSave = true   // history is opt-in: one tap explains it, turns it on and saves
                }
            }
        }
        Spacer(Modifier.height(10.dp))
        Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
            OutlineButton(stringResource(R.string.res_track), Modifier.weight(1f)) { onTrack() }
            OutlineButton(stringResource(R.string.res_retake), Modifier.weight(1f)) { onRetake() }
        }
        Spacer(Modifier.height(24.dp))
    }
    if (!com.skinnova.app.ui.LocalAppLocked.current && askSave) AlertDialog(onDismissRequest = { askSave = false },
        title = { Text(stringResource(R.string.res_save_title)) },
        text = { Text(stringResource(R.string.res_save_body)) },
        confirmButton = { TextButton({ askSave = false; vm.c.settings.setHistory(true); vm.save(); saved = true }) { Text(stringResource(R.string.res_save_ok)) } },
        dismissButton = { TextButton({ askSave = false }) { Text(stringResource(R.string.cancel)) } })
}

@Composable
private fun LikelihoodPill(l: String) {
    val sn = LocalSn.current
    val (txt, col) = when (l) { "higher" -> R.string.res_lk_higher to sn.accT; "possible" -> R.string.res_lk_possible to sn.pur; else -> R.string.res_lk_less_likely to sn.mut }
    Box(Modifier.heightIn(min = 24.dp).clip(RoundedCornerShape(12.dp)).border(1.dp, col, RoundedCornerShape(12.dp)).padding(horizontal = 10.dp, vertical = 3.dp)) {
        Text(stringResource(txt), style = SnType.micro, color = col)
    }
}

@Composable
internal fun BulletCard(title: String, items: List<String>, tts: Boolean, lang: String) {
    if (items.isEmpty()) return
    val sn = LocalSn.current
    SnCard(framed = false) {
        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(title, style = SnType.title, color = sn.ink, modifier = Modifier.weight(1f))
                if (tts) SpeakButton({ title + ". " + items.joinToString(". ") }, lang)
            }
            items.forEach { Row { Text("•  ", color = sn.accT, style = SnType.body); Text(it, style = SnType.body, color = sn.mut) } }
        }
    }
}

@Composable
private fun answerChips(r: FinalResult): List<String> {
    val a = r.answers
    val site = mapOf("face" to R.string.site_face, "scalp" to R.string.site_scalp, "neck" to R.string.site_neck, "chest" to R.string.site_chest,
        "back" to R.string.site_back, "abdomen" to R.string.site_abdomen, "arm" to R.string.site_arm, "hand" to R.string.site_hand,
        "leg" to R.string.site_leg, "foot" to R.string.site_foot, "groin" to R.string.site_groin, "nails" to R.string.site_nails)[a.bodySite] ?: R.string.site_other
    val dur = mapOf("lt_1w" to R.string.dur_lt_1w, "1_4w" to R.string.dur_1_4w, "1_6m" to R.string.dur_1_6m, "gt_6m" to R.string.dur_gt_6m)[a.duration]!!
    val lv = listOf(R.string.lvl_0, R.string.lvl_1, R.string.lvl_2, R.string.lvl_3)
    return listOf(stringResource(site), stringResource(dur), stringResource(R.string.chip_itch) + ": " + stringResource(lv[a.itch]),
        stringResource(R.string.chip_pain) + ": " + stringResource(lv[a.pain]))
}

/** While the model writes: if the phone may stop SkinNova in the background, offer the "keep running" exemption. */
@Composable
fun KeepRunningTip() {
    val sn = LocalSn.current
    val ctx = androidx.compose.ui.platform.LocalContext.current
    var exempt by remember { mutableStateOf(com.skinnova.app.notify.Background.exempt(ctx)) }
    val lifecycle = androidx.lifecycle.compose.LocalLifecycleOwner.current
    androidx.compose.runtime.DisposableEffect(lifecycle) {   // re-check when the user comes back from the settings screen
        val obs = androidx.lifecycle.LifecycleEventObserver { _, e -> if (e == androidx.lifecycle.Lifecycle.Event.ON_RESUME) exempt = com.skinnova.app.notify.Background.exempt(ctx) }
        lifecycle.lifecycle.addObserver(obs); onDispose { lifecycle.lifecycle.removeObserver(obs) }
    }
    if (exempt && !com.skinnova.app.notify.Background.aggressiveMaker) return
    Column(Modifier.fillMaxWidth().clip(RoundedCornerShape(18.dp)).background(sn.surf2).padding(12.dp)) {
        Text(stringResource(R.string.bg_title), style = SnType.label, color = sn.ink)
        Text(stringResource(if (com.skinnova.app.notify.Background.aggressiveMaker) R.string.bg_body_maker else R.string.bg_body), style = SnType.caption, color = sn.mut)
        if (!exempt) Box(Modifier.heightIn(min = 48.dp).clickable(role = Role.Button) { com.skinnova.app.notify.Background.request(ctx) }, contentAlignment = Alignment.CenterStart) {
            Text(stringResource(R.string.bg_allow) + "  →", style = SnType.label, color = sn.accT)
        }
    }
    Spacer(Modifier.height(10.dp))
}

/** Analyzing screen: what is already known while the model writes (deterministic, from strings.xml + the image model). */
@Composable
private fun EarlyLookCard(e: com.skinnova.app.ui.EarlyLook) {
    val sn = LocalSn.current
    Column(Modifier.fillMaxWidth()) {
        val msgs = e.ruleMessages.mapNotNull { k -> ruleMessageRes(k)?.let { if (k.startsWith("rf_t")) stringResource(it, "") else stringResource(it) } }
        if (msgs.isNotEmpty()) { RedFlagBanner(msgs); Spacer(Modifier.height(10.dp)) }
        SnCard(framed = false) {
            Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                Text(stringResource(R.string.an_early_title), style = SnType.title, color = sn.ink)
                Text(stringResource(R.string.an_early_sub), style = SnType.micro, color = sn.mut)
                e.top3.filter { it.p >= 0.05 }.forEachIndexed { i, s ->
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Text(stringResource(categoryNameRes(s.key)), style = SnType.label, color = sn.ink, modifier = Modifier.weight(1f))
                        LikelihoodPill(com.skinnova.app.ml.Fallback.likelihood(i, s.p))
                    }
                }
                Spacer(Modifier.height(4.dp))
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(stringResource(R.string.an_early_level), style = SnType.caption, color = sn.mut, modifier = Modifier.weight(1f))
                    Text(stringResource(when (e.tier) { Tier.LOW -> R.string.tier_LOW; Tier.MODERATE -> R.string.tier_MODERATE
                        Tier.HIGH -> R.string.tier_HIGH; Tier.URGENT -> R.string.tier_URGENT }), style = SnType.label, color = sn.tier(e.tier))
                }
            }
        }
        Spacer(Modifier.height(10.dp))
    }
}
