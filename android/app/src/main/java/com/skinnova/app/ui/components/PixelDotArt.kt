package com.skinnova.app.ui.components

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Paint
import androidx.annotation.DrawableRes
import androidx.compose.foundation.Canvas
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.produceState
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.FilterQuality
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.IntSize
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext
import kotlin.math.PI
import kotlin.math.sin

/** The dot-matrix art: small source bitmap (one pixel per cell) + where the stars are (for the twinkle). */
class DotArt(val image: ImageBitmap, val cell: Int, val gridW: Int, val gridH: Int, val stars: List<Star>)
data class Star(val x: Int, val y: Int, val color: Int, val phase: Float, val period: Float)

/**
 * Landing art as crisp "ASCII dots": each source pixel becomes a [cell]×[cell] square with a 1 px darker gap, drawn
 * once (off the main thread) at an INTEGER cell size so nothing is ever resampled (the old 1080 px art drawn at
 * ×1.015 smeared its fine dot texture = "blurry"). Stars twinkle at 15 fps; only those few cells are redrawn.
 */
suspend fun buildDotArt(ctx: android.content.Context, @DrawableRes res: Int, cell: Int): DotArt = withContext(Dispatchers.Default) {
    val src = BitmapFactory.decodeResource(ctx.resources, res, BitmapFactory.Options().apply { inScaled = false })
    val w = src.width; val h = src.height
    val px = IntArray(w * h).also { src.getPixels(it, 0, w, 0, 0, w, h) }
    val out = Bitmap.createBitmap(w * cell, h * cell, Bitmap.Config.ARGB_8888)
    val cv = Canvas(out); val p = Paint()
    val gap = if (cell >= 4) 1 else 0
    for (y in 0 until h) for (x in 0 until w) {
        val c = px[y * w + x]
        p.color = dim(c, 0.55f); cv.drawRect((x * cell).toFloat(), (y * cell).toFloat(), ((x + 1) * cell).toFloat(), ((y + 1) * cell).toFloat(), p)
        p.color = c; cv.drawRect((x * cell).toFloat(), (y * cell).toFloat(), ((x + 1) * cell - gap).toFloat(), ((y + 1) * cell - gap).toFloat(), p)
    }
    // stars: bright cells in the sky (upper 55 %) that stand out from their neighbours
    val rnd = java.util.Random(7); val stars = ArrayList<Star>()
    fun lum(i: Int) = ((i shr 16 and 255) + (i shr 8 and 255) + (i and 255)) / 3f
    for (y in 1 until (h * 0.55f).toInt()) for (x in 1 until w - 1) {
        val l = lum(px[y * w + x]); if (l < 110f) continue
        var around = 0f; for (dy in -1..1) for (dx in -1..1) if (dx != 0 || dy != 0) around += lum(px[(y + dy) * w + x + dx])
        if (l > around / 8f * 1.35f) stars += Star(x, y, px[y * w + x], rnd.nextFloat() * 2f * PI.toFloat(), 1.8f + rnd.nextFloat() * 2.4f)
    }
    src.recycle()
    DotArt(out.asImageBitmap(), cell, w, h, stars)
}

private fun dim(c: Int, f: Float): Int = (0xFF shl 24) or ((((c shr 16) and 255) * f).toInt() shl 16) or ((((c shr 8) and 255) * f).toInt() shl 8) or (((c and 255) * f).toInt())

/** Draws [art] at 1:1 pixels, top-left at the canvas origin, with stars twinkling at 15 fps. */
@Composable
fun DotArtCanvas(art: DotArt, modifier: Modifier = Modifier) {
    var frame by remember { mutableIntStateOf(0) }
    LaunchedEffect(art) { while (true) { delay(66); frame++ } }   // 15 fps, only the stars change
    Canvas(modifier) {
        drawImage(art.image, IntOffset.Zero, IntSize(art.image.width, art.image.height), filterQuality = FilterQuality.None)
        val t = frame / 15f; val c = art.cell.toFloat()
        art.stars.forEach { s ->
            val a = 0.5f + 0.5f * sin(2f * PI.toFloat() * t / s.period + s.phase)            // 0..1
            if (a > 0.55f) {   // brighten: a soft glow cell around the star, then the star itself
                drawRect(Color(s.color).copy(alpha = (a - 0.55f) * 0.5f), Offset((s.x - 1) * c, (s.y - 1) * c), Size(3 * c - 1, 3 * c - 1))
                drawRect(Color.White.copy(alpha = (a - 0.55f) * 1.2f), Offset(s.x * c, s.y * c), Size(c - 1, c - 1))
            } else drawRect(Color(0xFF0A0E1E).copy(alpha = (0.55f - a) * 0.9f), Offset(s.x * c, s.y * c), Size(c - 1, c - 1))   // fade
        }
    }
}
