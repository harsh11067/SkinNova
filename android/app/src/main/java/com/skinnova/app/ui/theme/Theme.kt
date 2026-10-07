package com.skinnova.app.ui.theme

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.Immutable
import androidx.compose.runtime.staticCompositionLocalOf
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.Font
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.skinnova.app.R
import com.skinnova.app.model.Tier

/** Tokens from design/tokens.json (official design THEMES). */
@Immutable
data class SnColors(
    val isDark: Boolean,
    val bg: Color, val bgBrush: Brush, val surf: Color, val surf2: Color, val ink: Color, val mut: Color, val title: Color,
    val line: Color, val line2: Color, val acc: Color, val accT: Color, val accSoft: Color, val glow: Color, val pur: Color,
    val track: Color, val btn: Color, val onAcc: Color, val nav: Color,
    val low: Color, val moderate: Color, val high: Color, val urgent: Color,
) {
    fun tier(t: Tier) = when (t) { Tier.LOW -> low; Tier.MODERATE -> moderate; Tier.HIGH -> high; Tier.URGENT -> urgent }
}

private fun c(hex: Long) = Color(hex)

val DarkSn = SnColors(
    isDark = true, bg = c(0xFF070B18), bgBrush = Brush.verticalGradient(listOf(c(0xFF070B18), c(0xFF070B18))),
    surf = c(0xFF0C1226), surf2 = c(0xFF141B3A), ink = c(0xFFEFE9F3), mut = c(0xFFAAA6CB), title = c(0xFFF6E0C8),
    line = c(0xFF55519A), line2 = c(0xFF232247), acc = c(0xFFEAB88C), accT = c(0xFFF4D3B2), accSoft = c(0x4DEAB88C),
    glow = c(0x38EAB88C), pur = c(0xFFA68BF0), track = c(0xFF24234A), btn = c(0xFF0D1120), onAcc = c(0xFF1B1222), nav = c(0xE00A0E1E),
    low = c(0xFF6FD39A), moderate = c(0xFFF2C46D), high = c(0xFFF39A5B), urgent = c(0xFFFF7A7A),
)

val LightSn = SnColors(
    isDark = false, bg = c(0xFFE0D6EC), bgBrush = Brush.verticalGradient(listOf(c(0xFFD6CBE9), c(0xFFE2D8EE), c(0xFFECDCD8))),
    surf = c(0x8FEEE5F9), surf2 = c(0x8CCDC0E6), ink = c(0xFF1C1640), mut = c(0xFF4A4376), title = c(0xFF2A1F5A),
    line = c(0xFF9A8ECB), line2 = c(0x528C7EC4), acc = c(0xFFA4572A), accT = c(0xFF7F3D17), accSoft = c(0x47A4572A),
    glow = c(0x4DD68C60), pur = c(0xFF4E3A9D), track = c(0x336E5EB0), btn = c(0xCCFCECE0), onAcc = c(0xFFFFF8F2), nav = c(0xC7E8E0F4),
    // tier colours are also text (triage title, tags): darkened to ≥ 4.6:1 on every light background incl. the red-flag tint
    low = c(0xFF165534), moderate = c(0xFF674300), high = c(0xFF7D3614), urgent = c(0xFF911F18),
)

val PlexMono = FontFamily(
    Font(R.font.plex_mono_regular, FontWeight.Normal),
    Font(R.font.plex_mono_medium, FontWeight.Medium),
    Font(R.font.plex_mono_semibold, FontWeight.SemiBold),
)
val Pixelify = FontFamily(Font(R.font.pixelify_sans, FontWeight.Normal))

object SnType {
    val micro = TextStyle(fontFamily = PlexMono, fontSize = 10.sp, lineHeight = 14.sp)
    val caption = TextStyle(fontFamily = PlexMono, fontSize = 10.5.sp, lineHeight = 16.sp)
    val body = TextStyle(fontFamily = PlexMono, fontSize = 11.5.sp, lineHeight = 18.sp)
    val bodyL = TextStyle(fontFamily = PlexMono, fontSize = 12.5.sp, lineHeight = 19.sp)
    val label = TextStyle(fontFamily = PlexMono, fontSize = 13.sp, lineHeight = 18.sp)
    val title = TextStyle(fontFamily = PlexMono, fontSize = 15.sp, lineHeight = 21.sp)
    val titleL = TextStyle(fontFamily = PlexMono, fontSize = 17.sp, lineHeight = 23.sp)
    val headline = TextStyle(fontFamily = PlexMono, fontSize = 20.sp, lineHeight = 26.sp)
    val display = TextStyle(fontFamily = PlexMono, fontSize = 21.sp, lineHeight = 28.sp)
    val pixel = TextStyle(fontFamily = Pixelify, fontSize = 17.sp, letterSpacing = 0.6.sp)
    val overline = TextStyle(fontFamily = PlexMono, fontSize = 11.sp, letterSpacing = 1.6.sp)
}

val LocalSn = staticCompositionLocalOf { DarkSn }

@Composable
fun SkinNovaTheme(dark: Boolean, content: @Composable () -> Unit) {
    val sn = if (dark) DarkSn else LightSn
    val scheme = if (dark) darkColorScheme(primary = sn.acc, onPrimary = sn.onAcc, background = sn.bg, surface = sn.surf, onSurface = sn.ink)
    else lightColorScheme(primary = sn.acc, onPrimary = sn.onAcc, background = sn.bg, surface = sn.bg, onSurface = sn.ink)
    CompositionLocalProvider(LocalSn provides sn) {
        MaterialTheme(colorScheme = scheme, content = content)
    }
}

/** The design's signature card: 1dp line border + inset 3dp surface + 1dp line2 ring, soft shadow. */
fun Modifier.snCard(sn: SnColors, radius: Dp = 24.dp, framed: Boolean = true, border: Color? = null): Modifier {
    val shape = RoundedCornerShape(radius)
    val base = this.shadow(if (framed) 10.dp else 0.dp, shape, ambientColor = Color.Black.copy(alpha = .35f), spotColor = Color.Black.copy(alpha = .35f))
        .clip(shape).background(sn.surf).border(1.dp, border ?: if (framed) sn.line else sn.line2, shape)
    return if (framed) base.padding(3.dp).border(1.dp, sn.line2, RoundedCornerShape(radius - 3.dp)) else base
}
