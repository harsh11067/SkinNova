package com.skinnova.app.ml

/**
 * Photo checks before classification (architecture §3): blur (variance of Laplacian on a 256-px grey image),
 * exposure (mean luma, clipped fraction), skin-pixel ratio (YCbCr box, Chai & Ngan), size.
 * Thresholds are tuned by ml/eval/quality_gate.py on val images (test.md C6: reject blur σ ≥ 3 at ≥ 90%).
 */
data class Quality(
    val blurVar: Double, val meanLuma: Double, val clippedDark: Double, val clippedBright: Double,
    val skinRatio: Double, val minSide: Int, val issues: List<Issue>,
) {
    enum class Issue { BLURRY, DARK, BRIGHT, SMALL, NO_SKIN, NOT_SKIN }   // NOT_SKIN: image model's skin-photo gate
    val ok get() = issues.isEmpty()
}

object QualityGate {
    // Tuned by ml/eval/quality_gate.py on 500 val photos (reports/quality_gate.json, 2026-10-06):
    // σ=3 blur rejected 98.8 %, clean false rejects 4.8 %; dark ×0.25 rejected 78 %.
    var BLUR_MIN = 34.8
    var LUMA_MIN = 41.0
    var LUMA_MAX = 227.5
    var SKIN_MIN = 0.038
    const val MIN_SIDE = 224

    fun check(img: Rgb): Quality {
        val small = downscaleGray(img, 256)
        val (gw, gh, g) = small
        var sum = 0.0; var dark = 0; var bright = 0
        for (v in g) { sum += v; if (v < 10) dark++; if (v > 245) bright++ }
        val mean = sum / g.size
        // Laplacian (4-neighbour) variance
        var s = 0.0; var s2 = 0.0; var n = 0
        for (y in 1 until gh - 1) for (x in 1 until gw - 1) {
            val i = y * gw + x
            val l = (g[i - 1] + g[i + 1] + g[i - gw] + g[i + gw] - 4 * g[i]).toDouble()
            s += l; s2 += l * l; n++
        }
        val lapVar = if (n > 0) s2 / n - (s / n) * (s / n) else 0.0
        val skin = skinRatio(img)
        val issues = buildList {
            if (minOf(img.w, img.h) < MIN_SIDE) add(Quality.Issue.SMALL)
            if (lapVar < BLUR_MIN) add(Quality.Issue.BLURRY)
            if (mean < LUMA_MIN) add(Quality.Issue.DARK)
            if (mean > LUMA_MAX) add(Quality.Issue.BRIGHT)
            if (skin < SKIN_MIN) add(Quality.Issue.NO_SKIN)
        }
        return Quality(lapVar, mean, dark.toDouble() / g.size, bright.toDouble() / g.size, skin, minOf(img.w, img.h), issues)
    }

    /** Box-average to ≤ maxSide, BT.601 luma. */
    fun downscaleGray(img: Rgb, maxSide: Int): Triple<Int, Int, IntArray> {
        val f = maxOf(1, (maxOf(img.w, img.h) + maxSide - 1) / maxSide)
        val w = img.w / f; val h = img.h / f
        val out = IntArray(w * h)
        for (y in 0 until h) for (x in 0 until w) {
            var acc = 0
            for (dy in 0 until f) for (dx in 0 until f) {
                val p = img.px[(y * f + dy) * img.w + x * f + dx]
                acc += (299 * ((p shr 16) and 0xFF) + 587 * ((p shr 8) and 0xFF) + 114 * (p and 0xFF)) / 1000
            }
            out[y * w + x] = acc / (f * f)
        }
        return Triple(w, h, out)
    }

    /** Fraction of pixels in the Cb∈[77,127], Cr∈[133,173] skin box — works across skin tones because it ignores luma. */
    fun skinRatio(img: Rgb): Double {
        val step = maxOf(1, (img.w * img.h) / 40_000)
        var hit = 0; var tot = 0
        var i = 0
        while (i < img.px.size) {
            val p = img.px[i]; val r = (p shr 16) and 0xFF; val g = (p shr 8) and 0xFF; val b = p and 0xFF
            val cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b
            val cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b
            if (cb in 77.0..127.0 && cr in 133.0..173.0) hit++
            tot++; i += step
        }
        return if (tot == 0) 0.0 else hit.toDouble() / tot
    }
}
