package com.skinnova.app.ui.components

import androidx.annotation.DrawableRes
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.rotate
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.FilterQuality
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.res.imageResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.skinnova.app.R
import com.skinnova.app.model.Tier
import com.skinnova.app.ui.theme.LocalSn
import com.skinnova.app.ui.theme.SnType
import com.skinnova.app.ui.theme.snCard
import androidx.compose.foundation.Canvas

/** Pixel-art image (design assets are pixel art: nearest-neighbour scaling, never smoothed). */
@Composable
fun PixelImage(@DrawableRes res: Int, modifier: Modifier = Modifier, scale: ContentScale = ContentScale.Crop) {
    Image(ImageBitmap.imageResource(res), contentDescription = null, modifier = modifier, contentScale = scale, filterQuality = FilterQuality.None)
}

@Composable
fun TopBar(title: String, onBack: () -> Unit, trailing: @Composable () -> Unit = {}) {
    val sn = LocalSn.current
    Row(Modifier.fillMaxWidth().height(48.dp), verticalAlignment = Alignment.CenterVertically) {
        Box(Modifier.size(48.dp).clip(RoundedCornerShape(16.dp)).clickable(role = Role.Button, onClick = onBack)
            .semantics { contentDescription = "Back" }, contentAlignment = Alignment.Center) {
            Text("←", color = sn.ink, fontSize = 20.sp)
        }
        Text(title, style = SnType.titleL, color = sn.ink, modifier = Modifier.weight(1f))
        trailing()
    }
}

@Composable
fun SnCard(modifier: Modifier = Modifier, framed: Boolean = true, border: Color? = null, radius: Dp = 24.dp, onClick: (() -> Unit)? = null,
           content: @Composable () -> Unit) {
    val sn = LocalSn.current
    var m = modifier.snCard(sn, radius, framed, border)
    if (onClick != null) m = m.clickable(role = Role.Button, onClick = onClick)
    Box(m.padding(if (framed) 15.dp else 18.dp)) { content() }
}

@Composable
fun PrimaryButton(text: String, modifier: Modifier = Modifier, enabled: Boolean = true, onClick: () -> Unit) {
    val sn = LocalSn.current
    Box(modifier.fillMaxWidth().heightIn(min = 54.dp).clip(RoundedCornerShape(20.dp))
        .background(if (enabled) sn.acc else sn.surf2).clickable(enabled = enabled, role = Role.Button, onClick = onClick),
        contentAlignment = Alignment.Center) {
        Text("$text  →", style = SnType.label.copy(fontWeight = FontWeight.Medium), color = if (enabled) sn.onAcc else sn.mut)
    }
}

@Composable
fun OutlineButton(text: String, modifier: Modifier = Modifier, onClick: () -> Unit) {
    val sn = LocalSn.current
    // clickable before the 3 dp inner ring: the whole 52 dp button is the touch target (it was 46 dp — A10)
    Box(modifier.heightIn(min = 52.dp).clip(RoundedCornerShape(18.dp)).clickable(role = Role.Button, onClick = onClick).background(sn.btn)
        .border(1.5.dp, sn.acc, RoundedCornerShape(18.dp)).padding(3.dp).border(1.dp, sn.accSoft, RoundedCornerShape(15.dp))
        .padding(horizontal = 14.dp), contentAlignment = Alignment.Center) {
        Text(text, style = SnType.bodyL, color = sn.accT, textAlign = TextAlign.Center)
    }
}

/** Option chip as in the design's question screen (46 dp pill; selected = accent border + glow). */
@Composable
fun Chip(label: String, selected: Boolean, onClick: () -> Unit, modifier: Modifier = Modifier) {
    val sn = LocalSn.current
    Box(modifier.heightIn(min = 48.dp).clip(RoundedCornerShape(23.dp))
        .background(if (selected) sn.surf2 else sn.surf)
        .border(if (selected) 1.5.dp else 1.dp, if (selected) sn.acc else sn.line2, RoundedCornerShape(23.dp))
        .clickable(role = Role.Button, onClick = onClick).padding(horizontal = 18.dp), contentAlignment = Alignment.Center) {
        Text(label, style = SnType.bodyL, color = if (selected) sn.accT else sn.ink)
    }
}

@Composable
fun SmallTag(text: String, color: Color? = null) {
    val sn = LocalSn.current
    Box(Modifier.height(26.dp).clip(RoundedCornerShape(13.dp)).background(sn.surf2).padding(horizontal = 11.dp), contentAlignment = Alignment.Center) {
        Text(text, style = SnType.micro, color = color ?: sn.ink, maxLines = 1)
    }
}

@Composable
fun Overline(text: String, modifier: Modifier = Modifier) {
    val sn = LocalSn.current
    // design touch: a 4 dp pixel marker in the accent colour before each section heading
    Row(modifier.padding(start = 4.dp, top = 22.dp, bottom = 10.dp), verticalAlignment = Alignment.CenterVertically) {
        Box(Modifier.size(4.dp).background(sn.accT.copy(alpha = .8f)))
        Spacer(Modifier.width(8.dp))
        Text(text, style = SnType.overline, color = sn.mut)
    }
}

@Composable
fun Bar(fraction: Float, color: Color? = null, modifier: Modifier = Modifier) {
    val sn = LocalSn.current
    Box(modifier.fillMaxWidth().height(6.dp).clip(RoundedCornerShape(3.dp)).background(sn.track)) {
        Box(Modifier.fillMaxWidth(fraction.coerceIn(0f, 1f)).height(6.dp).clip(RoundedCornerShape(3.dp)).background(color ?: sn.pur))
    }
}

@Composable
fun Disclaimer(modifier: Modifier = Modifier) {
    val sn = LocalSn.current
    Box(modifier.fillMaxWidth().clip(RoundedCornerShape(20.dp)).background(sn.surf2).padding(horizontal = 16.dp, vertical = 14.dp)) {
        Text(stringResource(R.string.disclaimer), style = SnType.caption, color = sn.mut)
    }
}

/** Triage: icon + words + colour (never colour alone — brief). Advice text comes from strings.xml, never from the LLM. */
@Composable
fun TriageCard(tier: Tier) {
    val sn = LocalSn.current
    val c = sn.tier(tier)
    val (label, advice, icon) = when (tier) {
        Tier.LOW -> Triple(R.string.tier_LOW, R.string.tier_adv_LOW, "✓")
        Tier.MODERATE -> Triple(R.string.tier_MODERATE, R.string.tier_adv_MODERATE, "◷")
        Tier.HIGH -> Triple(R.string.tier_HIGH, R.string.tier_adv_HIGH, "!")
        Tier.URGENT -> Triple(R.string.tier_URGENT, R.string.tier_adv_URGENT, "‼")
    }
    SnCard(border = c, framed = false) {
        Row(verticalAlignment = Alignment.Top) {
            Box(Modifier.size(34.dp).clip(RoundedCornerShape(11.dp)).background(c), contentAlignment = Alignment.Center) {
                Text(icon, color = sn.bg, fontSize = 16.sp, fontWeight = FontWeight.Bold)
            }
            Spacer(Modifier.width(12.dp))
            Column {
                Text(stringResource(label), style = SnType.title, color = c)
                Spacer(Modifier.height(6.dp))
                Text(stringResource(advice), style = SnType.body, color = sn.ink)
            }
        }
    }
}

@Composable
fun RedFlagBanner(messages: List<String>) {
    val sn = LocalSn.current
    Column(Modifier.fillMaxWidth().clip(RoundedCornerShape(20.dp)).background(sn.urgent.copy(alpha = .14f))
        .border(1.5.dp, sn.urgent, RoundedCornerShape(20.dp)).padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
        messages.forEach { m ->
            Row { Text("⚠ ", color = sn.urgent, style = SnType.label); Text(m, style = SnType.body, color = sn.ink) }
        }
    }
}

/** Design's spinning dotted ring (Analyzing / shutter). */
@Composable
fun SpinRing(size: Dp, color: Color, periodMs: Int, reverse: Boolean = false, dotted: Boolean = true, spinning: Boolean = true) {
    // spinning = false while the LLM runs on the GPU: a 60 fps animation competed with it for every frame
    val a = if (spinning) {
        val t = rememberInfiniteTransition(label = "spin")
        t.animateFloat(0f, if (reverse) -360f else 360f, infiniteRepeatable(tween(periodMs, easing = LinearEasing), RepeatMode.Restart), label = "a").value
    } else 0f
    Canvas(Modifier.size(size).rotate(a)) {
        drawCircle(color, style = Stroke(width = 1.5.dp.toPx(), pathEffect = PathEffect.dashPathEffect(if (dotted) floatArrayOf(2f, 6f) else floatArrayOf(10f, 8f))))
    }
}

@Composable
fun StatusPill(text: String, ok: Boolean) {
    val sn = LocalSn.current
    Row(Modifier.height(28.dp).clip(RoundedCornerShape(14.dp)).background(sn.surf).border(1.dp, sn.line2, RoundedCornerShape(14.dp))
        .padding(horizontal = 12.dp), verticalAlignment = Alignment.CenterVertically) {
        Box(Modifier.size(7.dp).clip(CircleShape).background(if (ok) Color(0xFF6FD39A) else sn.moderate))
        Spacer(Modifier.width(8.dp))
        Text(text, style = SnType.caption, color = sn.mut, maxLines = 1)
    }
}

@Composable
fun Toggle(on: Boolean) {
    val sn = LocalSn.current
    Box(Modifier.size(46.dp, 28.dp).clip(RoundedCornerShape(14.dp)).background(if (on) sn.acc else sn.track), contentAlignment = if (on) Alignment.CenterEnd else Alignment.CenterStart) {
        Box(Modifier.padding(3.dp).size(22.dp).clip(CircleShape).background(if (on) sn.onAcc else sn.mut))
    }
}

fun categoryNameRes(key: String): Int = when (key) {
    "eczema_atopic" -> R.string.cat_eczema_atopic
    "contact_dermatitis" -> R.string.cat_contact_dermatitis
    "seborrheic_dermatitis" -> R.string.cat_seborrheic_dermatitis
    "tinea" -> R.string.cat_tinea
    "scabies" -> R.string.cat_scabies
    "acne" -> R.string.cat_acne
    "psoriasis" -> R.string.cat_psoriasis
    "vitiligo" -> R.string.cat_vitiligo
    "benign_lesion" -> R.string.cat_benign_lesion
    "suspicious_lesion" -> R.string.cat_suspicious_lesion
    else -> R.string.cat_other
}

fun ruleMessageRes(key: String): Int? = when (key) {
    "rf_r1" -> R.string.rf_r1; "rf_r2" -> R.string.rf_r2; "rf_r3" -> R.string.rf_r3; "rf_r4" -> R.string.rf_r4
    "rf_r5" -> R.string.rf_r5; "rf_r6" -> R.string.rf_r6; "rf_r7" -> R.string.rf_r7; "rf_r8" -> R.string.rf_r8
    "rf_t1" -> R.string.rf_t1; "rf_t2" -> R.string.rf_t2; "rf_t3" -> R.string.rf_t3
    else -> null
}

/** Library thumbnail per category (design assets px_lib0..3; lesion classes use the journey thumbnails). */
fun categoryArt(key: String): Int = when (key) {
    "acne" -> R.drawable.px_lib0
    "eczema_atopic", "seborrheic_dermatitis", "psoriasis" -> R.drawable.px_lib1
    "tinea", "scabies" -> R.drawable.px_lib2
    "contact_dermatitis", "vitiligo" -> R.drawable.px_lib3
    "benign_lesion", "suspicious_lesion" -> R.drawable.px_j0
    else -> R.drawable.px_j1
}

fun siteNameRes(key: String?): Int = mapOf("face" to R.string.site_face, "scalp" to R.string.site_scalp, "neck" to R.string.site_neck,
    "chest" to R.string.site_chest, "back" to R.string.site_back, "abdomen" to R.string.site_abdomen, "arm" to R.string.site_arm,
    "hand" to R.string.site_hand, "leg" to R.string.site_leg, "foot" to R.string.site_foot, "groin" to R.string.site_groin,
    "nails" to R.string.site_nails)[key] ?: R.string.site_other

/** 48 dp read-aloud button (system TTS, i18n/Tts.kt); hidden when "Read results aloud" is off in Profile. */
@Composable
fun SpeakButton(text: () -> String, lang: String) {
    val ctx = androidx.compose.ui.platform.LocalContext.current
    val label = stringResource(R.string.res_read_aloud)
    Box(Modifier.size(48.dp).semantics { contentDescription = label }.clickable(role = Role.Button) {
        com.skinnova.app.i18n.Tts.speak(ctx, text(), lang)
    }, contentAlignment = Alignment.Center) { Text("🔊", fontSize = 16.sp) }
}

/** Real example photos for the Skin Library (ml/data/library_examples.py; SCIN + PAD-UFES-20, CC BY 4.0 — credits in
 *  assets/library/credits.json). Several skin tones per condition; empty for "other". */
fun categoryPhotos(key: String): List<Int> = when (key) {
    "eczema_atopic" -> listOf(R.drawable.lib_eczema_atopic_0, R.drawable.lib_eczema_atopic_1, R.drawable.lib_eczema_atopic_2)
    "contact_dermatitis" -> listOf(R.drawable.lib_contact_dermatitis_0, R.drawable.lib_contact_dermatitis_1, R.drawable.lib_contact_dermatitis_2)
    "acne" -> listOf(R.drawable.lib_acne_0, R.drawable.lib_acne_1, R.drawable.lib_acne_2)
    "tinea" -> listOf(R.drawable.lib_tinea_0, R.drawable.lib_tinea_1, R.drawable.lib_tinea_2)
    "psoriasis" -> listOf(R.drawable.lib_psoriasis_0, R.drawable.lib_psoriasis_1, R.drawable.lib_psoriasis_2)
    "scabies" -> listOf(R.drawable.lib_scabies_0, R.drawable.lib_scabies_1, R.drawable.lib_scabies_2)
    "seborrheic_dermatitis" -> listOf(R.drawable.lib_seborrheic_dermatitis_0, R.drawable.lib_seborrheic_dermatitis_1)
    "benign_lesion" -> listOf(R.drawable.lib_benign_lesion_0, R.drawable.lib_benign_lesion_1, R.drawable.lib_benign_lesion_2)
    "suspicious_lesion" -> listOf(R.drawable.lib_suspicious_lesion_0, R.drawable.lib_suspicious_lesion_1, R.drawable.lib_suspicious_lesion_2)
    "vitiligo" -> listOf(R.drawable.lib_vitiligo_0)
    else -> emptyList()
}

/** Single-line text-field style: one 48 sp line with the text centred, so the field's own tap target is ≥ 48 dp (A10). */
val fieldTextStyle get() = SnType.bodyL.copy(lineHeight = 48.sp,
    lineHeightStyle = androidx.compose.ui.text.style.LineHeightStyle(androidx.compose.ui.text.style.LineHeightStyle.Alignment.Center,
        androidx.compose.ui.text.style.LineHeightStyle.Trim.None))

/**
 * Pixel progress bar in the landing page's language (24 square segments in a framed track). [fraction] null =
 * indeterminate: a 3-segment highlight walks along at ~8 steps/s (one cheap redraw per step, no per-frame animation).
 */
@Composable
fun PixelProgressBar(fraction: Float?, modifier: Modifier = Modifier, segments: Int = 24) {
    val sn = LocalSn.current
    var step by androidx.compose.runtime.remember { androidx.compose.runtime.mutableIntStateOf(0) }
    if (fraction == null) androidx.compose.runtime.LaunchedEffect(Unit) { while (true) { kotlinx.coroutines.delay(120); step++ } }
    val lit = fraction?.let { (it.coerceIn(0f, 1f) * segments).toInt() }
    Row(modifier.fillMaxWidth().clip(RoundedCornerShape(8.dp)).background(sn.surf2).border(1.dp, sn.line2, RoundedCornerShape(8.dp)).padding(4.dp),
        horizontalArrangement = Arrangement.spacedBy(2.dp)) {
        repeat(segments) { i ->
            val on = if (lit != null) i < lit else ((i - step % (segments + 3)) in -2..0)
            Box(Modifier.weight(1f).height(8.dp).clip(RoundedCornerShape(1.dp)).background(if (on) sn.acc else sn.track))
        }
    }
}
