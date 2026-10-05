package com.skinnova.app.ml

import kotlin.math.abs
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin

/**
 * Port of Pillow's Resample.c (ImagingResample, 8-bit, fixed point) so on-device preprocessing equals the Python
 * training/eval preprocessing (test.md C4: most on-device accuracy bugs are resize-filter mismatches).
 * Images are packed RGB in an IntArray of 0xRRGGBB (alpha ignored).
 */
class Rgb(val w: Int, val h: Int, val px: IntArray) {
    fun crop(x0: Int, y0: Int, x1: Int, y1: Int): Rgb {
        val cw = x1 - x0; val ch = y1 - y0
        val out = IntArray(cw * ch)
        for (y in 0 until ch) System.arraycopy(px, (y + y0) * w + x0, out, y * cw, cw)
        return Rgb(cw, ch, out)
    }
}

object ImageOps {
    private const val PRECISION_BITS = 32 - 8 - 2

    enum class Filter(val support: Double) {
        BILINEAR(1.0), LANCZOS(3.0);
        fun f(x0: Double): Double = when (this) {
            BILINEAR -> { val x = abs(x0); if (x < 1.0) 1.0 - x else 0.0 }
            LANCZOS -> if (-3.0 <= x0 && x0 < 3.0) sinc(x0) * sinc(x0 / 3.0) else 0.0
        }
        private fun sinc(x: Double): Double { if (x == 0.0) return 1.0; val y = x * Math.PI; return sin(y) / y }
    }

    private class Coeffs(val bounds: IntArray, val k: IntArray, val ksize: Int)

    private fun precompute(inSize: Int, outSize: Int, filter: Filter): Coeffs {
        val scale = inSize.toDouble() / outSize
        val filterscale = max(scale, 1.0)
        val support = filter.support * filterscale
        val ksize = Math.ceil(support).toInt() * 2 + 1
        val kk = DoubleArray(outSize * ksize)
        val bounds = IntArray(outSize * 2)
        for (xx in 0 until outSize) {
            val center = (xx + 0.5) * scale
            val ss = 1.0 / filterscale
            var xmin = (center - support + 0.5).toInt(); if (xmin < 0) xmin = 0
            var xmax = (center + support + 0.5).toInt(); if (xmax > inSize) xmax = inSize
            xmax -= xmin
            var ww = 0.0
            for (x in 0 until xmax) { val w = filter.f((x + xmin - center + 0.5) * ss); kk[xx * ksize + x] = w; ww += w }
            for (x in 0 until xmax) if (ww != 0.0) kk[xx * ksize + x] /= ww
            for (x in xmax until ksize) kk[xx * ksize + x] = 0.0
            bounds[xx * 2] = xmin; bounds[xx * 2 + 1] = xmax
        }
        // normalize_coeffs_8bpc
        val k = IntArray(kk.size) { i -> val v = kk[i]; if (v < 0) (-0.5 + v * (1 shl PRECISION_BITS)).toInt() else (0.5 + v * (1 shl PRECISION_BITS)).toInt() }
        return Coeffs(bounds, k, ksize)
    }

    private fun clip8(v: Long): Int { val r = (v shr PRECISION_BITS).toInt(); return if (r < 0) 0 else if (r > 255) 255 else r }

    private fun horizontal(src: Rgb, outW: Int, filter: Filter): Rgb {
        val c = precompute(src.w, outW, filter)
        val out = IntArray(outW * src.h)
        val init = 1L shl (PRECISION_BITS - 1)
        for (y in 0 until src.h) {
            val row = y * src.w
            for (xx in 0 until outW) {
                val xmin = c.bounds[xx * 2]; val xmax = c.bounds[xx * 2 + 1]; val ko = xx * c.ksize
                var r = init; var g = init; var b = init
                for (x in 0 until xmax) {
                    val p = src.px[row + x + xmin]; val kv = c.k[ko + x].toLong()
                    r += ((p shr 16) and 0xFF) * kv; g += ((p shr 8) and 0xFF) * kv; b += (p and 0xFF) * kv
                }
                out[y * outW + xx] = (clip8(r) shl 16) or (clip8(g) shl 8) or clip8(b)
            }
        }
        return Rgb(outW, src.h, out)
    }

    private fun vertical(src: Rgb, outH: Int, filter: Filter): Rgb {
        val c = precompute(src.h, outH, filter)
        val out = IntArray(src.w * outH)
        val init = 1L shl (PRECISION_BITS - 1)
        for (yy in 0 until outH) {
            val ymin = c.bounds[yy * 2]; val ymax = c.bounds[yy * 2 + 1]; val ko = yy * c.ksize
            for (x in 0 until src.w) {
                var r = init; var g = init; var b = init
                for (y in 0 until ymax) {
                    val p = src.px[(y + ymin) * src.w + x]; val kv = c.k[ko + y].toLong()
                    r += ((p shr 16) and 0xFF) * kv; g += ((p shr 8) and 0xFF) * kv; b += (p and 0xFF) * kv
                }
                out[yy * src.w + x] = (clip8(r) shl 16) or (clip8(g) shl 8) or clip8(b)
            }
        }
        return Rgb(src.w, outH, out)
    }

    /** PIL Image.resize((w, h), filter): horizontal pass first, then vertical (Resample.c ImagingResampleInner). */
    fun resize(src: Rgb, w: Int, h: Int, filter: Filter): Rgb {
        var img = src
        if (w != img.w) img = horizontal(img, w, filter)
        if (h != img.h) img = vertical(img, h, filter)
        return img
    }

    /** ml/data/normalize.py: long side → 512 with LANCZOS if larger (round half to even like Python round()). */
    fun normalizeLongSide(src: Rgb, longSide: Int = 512): Rgb {
        val s = longSide.toDouble() / max(src.w, src.h)
        if (s >= 1) return src
        return resize(src, pyRound(src.w * s), pyRound(src.h * s), Filter.LANCZOS)
    }

    fun pyRound(x: Double): Int { val f = floor(x); val d = x - f; return (if (d > 0.5 || (d == 0.5 && f.toLong() % 2L != 0L)) f + 1 else f).toInt() }

    /** ml/cv/dataset.py center_square + Resize((384,384), BILINEAR). */
    fun cvInput(src: Rgb, size: Int = 384): Rgb {
        val s = min(src.w, src.h)
        val x0 = (src.w - s) / 2; val y0 = (src.h - s) / 2
        return resize(src.crop(x0, y0, x0 + s, y0 + s), size, size, Filter.BILINEAR)
    }
}
