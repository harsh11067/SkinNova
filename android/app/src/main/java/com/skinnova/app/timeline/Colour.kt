package com.skinnova.app.timeline

import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.cbrt
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.hypot
import kotlin.math.pow
import kotlin.math.sin
import kotlin.math.sqrt

/** CIEDE2000 + sRGB→Lab (D65). Python twin ml/timeline/colour.py; fixture tests/fixtures/ciede2000_cases.json. */
object Colour {
    private fun rad(d: Double) = d * PI / 180.0
    private fun deg(r: Double) = r * 180.0 / PI

    fun srgbToLab(r: Int, g: Int, b: Int): DoubleArray {
        fun lin(c: Int): Double { val v = c / 255.0; return if (v <= 0.04045) v / 12.92 else ((v + 0.055) / 1.055).pow(2.4) }
        val R = lin(r); val G = lin(g); val B = lin(b)
        val x = (0.4124564 * R + 0.3575761 * G + 0.1804375 * B) / 0.95047
        val y = (0.2126729 * R + 0.7151522 * G + 0.0721750 * B) / 1.0
        val z = (0.0193339 * R + 0.1191920 * G + 0.9503041 * B) / 1.08883
        val e = (6.0 / 29).pow(3)
        fun f(t: Double) = if (t > e) cbrt(t) else t / (3 * (6.0 / 29).pow(2)) + 4.0 / 29
        return doubleArrayOf(116 * f(y) - 16, 500 * (f(x) - f(y)), 200 * (f(y) - f(z)))
    }

    fun ciede2000(l1: DoubleArray, l2: DoubleArray): Double {
        val (L1, a1, b1) = Triple(l1[0], l1[1], l1[2]); val (L2, a2, b2) = Triple(l2[0], l2[1], l2[2])
        val C1 = hypot(a1, b1); val C2 = hypot(a2, b2); val Cb = (C1 + C2) / 2
        val G = 0.5 * (1 - sqrt(Cb.pow(7) / (Cb.pow(7) + 25.0.pow(7))))
        val a1p = (1 + G) * a1; val a2p = (1 + G) * a2
        val C1p = hypot(a1p, b1); val C2p = hypot(a2p, b2)
        fun hue(a: Double, b: Double) = if (a == 0.0 && b == 0.0) 0.0 else (deg(atan2(b, a)) % 360 + 360) % 360
        val h1p = hue(a1p, b1); val h2p = hue(a2p, b2)
        val dLp = L2 - L1; val dCp = C2p - C1p
        var dhp = 0.0
        if (C1p * C2p != 0.0) { dhp = h2p - h1p; if (dhp > 180) dhp -= 360 else if (dhp < -180) dhp += 360 }
        val dHp = 2 * sqrt(C1p * C2p) * sin(rad(dhp / 2))
        val Lbp = (L1 + L2) / 2; val Cbp = (C1p + C2p) / 2
        val hbp = when {
            C1p * C2p == 0.0 -> h1p + h2p
            abs(h1p - h2p) <= 180 -> (h1p + h2p) / 2
            h1p + h2p < 360 -> (h1p + h2p + 360) / 2
            else -> (h1p + h2p - 360) / 2
        }
        val T = 1 - 0.17 * cos(rad(hbp - 30)) + 0.24 * cos(rad(2 * hbp)) + 0.32 * cos(rad(3 * hbp + 6)) - 0.20 * cos(rad(4 * hbp - 63))
        val dTheta = 30 * exp(-((hbp - 275) / 25).pow(2))
        val Rc = 2 * sqrt(Cbp.pow(7) / (Cbp.pow(7) + 25.0.pow(7)))
        val Sl = 1 + 0.015 * (Lbp - 50).pow(2) / sqrt(20 + (Lbp - 50).pow(2))
        val Sc = 1 + 0.045 * Cbp; val Sh = 1 + 0.015 * Cbp * T
        val Rt = -sin(rad(2 * dTheta)) * Rc
        return sqrt((dLp / Sl).pow(2) + (dCp / Sc).pow(2) + (dHp / Sh).pow(2) + Rt * (dCp / Sc) * (dHp / Sh))
    }

    /** Jensen–Shannon divergence (base 2) of two probability vectors. */
    fun jsDivergence(p0: DoubleArray, q0: DoubleArray): Double {
        val p = p0.map { it + 1e-12 }.let { l -> val s = l.sum(); l.map { it / s } }
        val q = q0.map { it + 1e-12 }.let { l -> val s = l.sum(); l.map { it / s } }
        var d = 0.0
        for (i in p.indices) { val m = (p[i] + q[i]) / 2; d += 0.5 * p[i] * ln2(p[i] / m) + 0.5 * q[i] * ln2(q[i] / m) }
        return d
    }
    private fun ln2(x: Double) = kotlin.math.ln(x) / kotlin.math.ln(2.0)
}
