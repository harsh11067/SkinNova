package com.skinnova.app.timeline

import android.graphics.Bitmap
import com.skinnova.app.model.SnJson
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.calib3d.Calib3d
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.DMatch
import org.opencv.core.Mat
import org.opencv.core.MatOfDMatch
import org.opencv.core.MatOfKeyPoint
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.features2d.BFMatcher
import org.opencv.features2d.ORB
import org.opencv.imgproc.Imgproc
import kotlin.math.PI
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.sqrt

/**
 * SkinTimeline (USP-1, architecture §7). Twin of ml/timeline/metrics.py — same OpenCV calls & parameters.
 * Evaluated on synthetic ground truth by ml/timeline/eval_timeline.py (reports/timeline_eval.json).
 */
@Serializable
data class NoiseFloor(val area: Double, val contrast: Double, val n: Int)

@Serializable
data class ChangeMetrics(
    @SerialName("capture_id") val captureId: String = "",
    @SerialName("baseline_capture_id") val baselineCaptureId: String = "",
    @SerialName("days_since_baseline") val daysSinceBaseline: Int = 0,
    @SerialName("align_score") val alignScore: Double,
    @SerialName("align_inliers") val alignInliers: Int,
    @SerialName("align_ok") val alignOk: Boolean,
    @SerialName("coin_in_both") val coinInBoth: Boolean = false,
    @SerialName("area_ratio") val areaRatio: Double? = null,
    @SerialName("contrast_delta") val contrastDelta: Double? = null,
    @SerialName("border_irregularity_delta") val borderIrregularityDelta: Double? = null,
    @SerialName("cv_shift_js") val cvShiftJs: Double? = null,
    @SerialName("noise_floor") val noiseFloor: NoiseFloor? = null,
    val confidence: String = "low",
    @SerialName("lesion_type") val lesionType: Boolean = true,
    @SerialName("timeline_tier") val timelineTier: String = "LOW",
    @SerialName("triggered_rules") val triggeredRules: List<String> = emptyList(),
    @SerialName("area_raw") val areaRaw: Double? = null,
    @SerialName("contrast_raw") val contrastRaw: Double? = null,
) {
    fun json() = SnJson.encodeToString(serializer(), this)
}

object Timeline {
    const val ALIGN_MIN_RATIO = 0.25
    const val ALIGN_MIN_INLIERS = 30
    val ready: Boolean by lazy { OpenCVLoader.initLocal() }

    fun toRgbMat(b: Bitmap): Mat {
        val rgba = Mat(); Utils.bitmapToMat(b.copy(Bitmap.Config.ARGB_8888, false), rgba)
        val rgb = Mat(); Imgproc.cvtColor(rgba, rgb, Imgproc.COLOR_RGBA2RGB); rgba.release(); return rgb
    }

    fun resizeMax(m: Mat, side: Int): Mat {
        val s = side.toDouble() / max(m.cols(), m.rows())
        if (s >= 1) return m
        val o = Mat(); Imgproc.resize(m, o, Size(m.cols() * s, m.rows() * s), 0.0, 0.0, Imgproc.INTER_AREA); return o
    }

    data class Alignment(val h: Mat?, val ratio: Double, val inliers: Int) { val ok get() = h != null && ratio >= ALIGN_MIN_RATIO && inliers >= ALIGN_MIN_INLIERS }

    /** H maps new → base. */
    fun align(baseRgb: Mat, newRgb: Mat): Alignment {
        Core.setRNGSeed(0)
        val g1 = Mat(); val g2 = Mat()
        Imgproc.cvtColor(baseRgb, g1, Imgproc.COLOR_RGB2GRAY); Imgproc.cvtColor(newRgb, g2, Imgproc.COLOR_RGB2GRAY)
        val orb = ORB.create(1000)
        val k1 = MatOfKeyPoint(); val k2 = MatOfKeyPoint(); val d1 = Mat(); val d2 = Mat()
        orb.detectAndCompute(g1, Mat(), k1, d1); orb.detectAndCompute(g2, Mat(), k2, d2)
        val kp1 = k1.toArray(); val kp2 = k2.toArray()
        if (d1.empty() || d2.empty() || kp1.size < 10 || kp2.size < 10) return Alignment(null, 0.0, 0)
        val knn = ArrayList<MatOfDMatch>()
        BFMatcher.create(Core.NORM_HAMMING, false).knnMatch(d2, d1, knn, 2)
        val good = ArrayList<DMatch>()
        for (m in knn) { val a = m.toArray(); if (a.size == 2 && a[0].distance < 0.75f * a[1].distance) good += a[0] }
        if (good.size < 8) return Alignment(null, 0.0, 0)
        val src = MatOfPoint2f(*good.map { kp2[it.queryIdx].pt }.toTypedArray())
        val dst = MatOfPoint2f(*good.map { kp1[it.trainIdx].pt }.toTypedArray())
        val mask = Mat()
        val h = Calib3d.findHomography(src, dst, Calib3d.RANSAC, 4.0, mask)
        if (h.empty()) return Alignment(null, 0.0, 0)
        val n = Core.countNonZero(mask)
        return Alignment(h, n.toDouble() / good.size, n)
    }

    /** GrabCut seeded at the tap (normalised x,y). Returns CV_8U 0/1 mask. */
    fun segment(rgb: Mat, seedX: Double, seedY: Double, rFrac: Double = 0.12, iters: Int = 5): Mat {
        Core.setRNGSeed(0)
        val w = rgb.cols(); val h = rgb.rows()
        val sx = (seedX * w).toInt(); val sy = (seedY * h).toInt()
        val r = rFrac * min(w, h)
        val mask = Mat(h, w, CvType.CV_8U, Scalar(Imgproc.GC_PR_BGD.toDouble()))
        Imgproc.circle(mask, Point(sx.toDouble(), sy.toDouble()), (1.6 * r).toInt(), Scalar(Imgproc.GC_PR_FGD.toDouble()), -1)
        Imgproc.circle(mask, Point(sx.toDouble(), sy.toDouble()), max(2, (r / 3).toInt()), Scalar(Imgproc.GC_FGD.toDouble()), -1)
        val b = max(2, (0.02 * min(w, h)).toInt())
        Imgproc.rectangle(mask, Point(0.0, 0.0), Point((w - 1).toDouble(), (b - 1).toDouble()), Scalar(Imgproc.GC_BGD.toDouble()), -1)
        Imgproc.rectangle(mask, Point(0.0, (h - b).toDouble()), Point((w - 1).toDouble(), (h - 1).toDouble()), Scalar(Imgproc.GC_BGD.toDouble()), -1)
        Imgproc.rectangle(mask, Point(0.0, 0.0), Point((b - 1).toDouble(), (h - 1).toDouble()), Scalar(Imgproc.GC_BGD.toDouble()), -1)
        Imgproc.rectangle(mask, Point((w - b).toDouble(), 0.0), Point((w - 1).toDouble(), (h - 1).toDouble()), Scalar(Imgproc.GC_BGD.toDouble()), -1)
        val bgr = Mat(); Imgproc.cvtColor(rgb, bgr, Imgproc.COLOR_RGB2BGR)
        Imgproc.grabCut(bgr, mask, Rect(), Mat(), Mat(), iters, Imgproc.GC_INIT_WITH_MASK)
        // fg = (mask == FGD) | (mask == PR_FGD), as 0/1
        val f1 = Mat(); val f3 = Mat(); val fg = Mat()
        Core.compare(mask, Scalar(Imgproc.GC_FGD.toDouble()), f1, Core.CMP_EQ)
        Core.compare(mask, Scalar(Imgproc.GC_PR_FGD.toDouble()), f3, Core.CMP_EQ)
        Core.bitwise_or(f1, f3, fg); Core.divide(fg, Scalar(255.0), fg)
        val k = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(5.0, 5.0))
        Imgproc.morphologyEx(fg, fg, Imgproc.MORPH_OPEN, k); Imgproc.morphologyEx(fg, fg, Imgproc.MORPH_CLOSE, k)
        val labels = Mat(); val n = Imgproc.connectedComponents(fg, labels)
        var keep = labels.get(min(h - 1, sy), min(w - 1, sx))[0].toInt()
        if (keep == 0) {
            if (n <= 1) return fg
            var best = 0; var bestN = -1
            for (i in 1 until n) { val c = Mat(); Core.compare(labels, Scalar(i.toDouble()), c, Core.CMP_EQ); val cnt = Core.countNonZero(c); if (cnt > bestN) { bestN = cnt; best = i } }
            keep = best
        }
        val out = Mat(); Core.compare(labels, Scalar(keep.toDouble()), out, Core.CMP_EQ)
        Core.divide(out, Scalar(255.0), out)
        return out
    }

    /** (cx, cy, r) of a coin not overlapping the lesion, or null. */
    fun detectCoin(rgb: Mat, lesion: Mat? = null): DoubleArray? {
        val g = Mat(); Imgproc.cvtColor(rgb, g, Imgproc.COLOR_RGB2GRAY); Imgproc.medianBlur(g, g, 5)
        val m = min(g.cols(), g.rows())
        val c = Mat()
        Imgproc.HoughCircles(g, c, Imgproc.HOUGH_GRADIENT, 1.2, m * 0.2, 120.0, 40.0, (0.02 * m).toInt(), (0.15 * m).toInt())
        for (i in 0 until c.cols()) {
            val v = c.get(0, i)
            if (lesion != null) {
                val disc = Mat.zeros(lesion.size(), CvType.CV_8U)
                Imgproc.circle(disc, Point(v[0], v[1]), v[2].toInt(), Scalar(1.0), -1)
                val inter = Mat(); Core.bitwise_and(disc, lesion, inter)
                if (Core.countNonZero(inter) > 0.1 * Core.countNonZero(disc)) continue
            }
            return v
        }
        return null
    }

    private fun meanLab(rgb: Mat, mask: Mat): DoubleArray? {
        val w = rgb.cols(); val h = rgb.rows()
        val px = ByteArray(w * h * 3); rgb.get(0, 0, px)
        val mk = ByteArray(w * h); mask.get(0, 0, mk)
        var L = 0.0; var A = 0.0; var B = 0.0; var n = 0
        for (i in 0 until w * h) if (mk[i].toInt() != 0) {
            val lab = Colour.srgbToLab(px[3 * i].toInt() and 255, px[3 * i + 1].toInt() and 255, px[3 * i + 2].toInt() and 255)
            L += lab[0]; A += lab[1]; B += lab[2]; n++
        }
        return if (n == 0) null else doubleArrayOf(L / n, A / n, B / n)
    }

    fun ring(mask: Mat): Mat {
        val a = max(1.0, sqrt(Core.countNonZero(mask) / PI))
        val ks = 2 * (0.6 * a).toInt() + 1
        val d = Mat(); Imgproc.dilate(mask, d, Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(ks.toDouble(), ks.toDouble())))
        val inv = Mat(); Core.subtract(Mat.ones(mask.size(), CvType.CV_8U), mask, inv)
        val r = Mat(); Core.bitwise_and(d, inv, r); return r
    }

    fun lesionContrast(rgb: Mat, mask: Mat): Double {
        val l = meanLab(rgb, mask) ?: return 0.0
        val r = meanLab(rgb, ring(mask)) ?: return 0.0
        return Colour.ciede2000(l, r)
    }

    fun irregularity(mask: Mat): Double {
        val cs = ArrayList<MatOfPoint>(); Imgproc.findContours(mask.clone(), cs, Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE)
        val c = cs.maxByOrNull { Imgproc.contourArea(it) } ?: return 0.0
        val a = Imgproc.contourArea(c); if (a <= 0) return 0.0
        return Imgproc.arcLength(MatOfPoint2f(*c.toArray()), true).pow(2) / (4 * PI * a)
    }

    /** contracts §8 metrics; tier/rules filled by the caller via RedFlagRules.applyTimeline. */
    fun compute(base: Bitmap, new: Bitmap, seedX: Double, seedY: Double, coinMm: Double?, cvBase: DoubleArray?, cvNew: DoubleArray?,
                noise: NoiseFloor?, lesionType: Boolean): ChangeMetrics {
        check(ready) { "OpenCV failed to load" }
        val b = resizeMax(toRgbMat(base), 512); val n = resizeMax(toRgbMat(new), 512)
        val al = align(b, n)
        if (!al.ok) return ChangeMetrics(alignScore = al.ratio, alignInliers = al.inliers, alignOk = false, lesionType = lesionType, noiseFloor = noise)
        val warped = Mat(); Imgproc.warpPerspective(n, warped, al.h, b.size(), Imgproc.INTER_LINEAR, Core.BORDER_REFLECT)
        val m0 = segment(b, seedX, seedY); val m1 = segment(warped, seedX, seedY)
        val a0 = Core.countNonZero(m0).toDouble(); val a1 = Core.countNonZero(m1).toDouble()
        val c0 = detectCoin(b, m0); val c1 = detectCoin(n)
        val coinBoth = c0 != null && c1 != null && coinMm != null
        val areaRatio = if (coinBoth) {
            val hInv = al.h!!.inv(); val m1own = Mat(); Imgproc.warpPerspective(m1, m1own, hInv, n.size(), Imgproc.INTER_NEAREST)
            val mm0 = a0 * (coinMm!! / (2 * c0!![2])).pow(2); val mm1 = Core.countNonZero(m1own) * (coinMm / (2 * c1!![2])).pow(2)
            if (mm0 > 0) mm1 / mm0 else null
        } else if (a0 > 0) a1 / a0 else null
        val cd = lesionContrast(warped, m1) - lesionContrast(b, m0)
        val conf = if (coinBoth && (noise?.n ?: 0) >= 3) "ok" else "low"
        return ChangeMetrics(alignScore = al.ratio, alignInliers = al.inliers, alignOk = true, coinInBoth = coinBoth, areaRatio = areaRatio,
            contrastDelta = cd, borderIrregularityDelta = irregularity(m1) - irregularity(m0),
            cvShiftJs = if (cvBase != null && cvNew != null) Colour.jsDivergence(cvBase, cvNew) else null,
            noiseFloor = noise, confidence = conf, lesionType = lesionType)
    }

    /** Lesion mask overlay for the UI (seed preview). */
    fun maskBitmap(photo: Bitmap, seedX: Double, seedY: Double): Bitmap {
        val m = resizeMax(toRgbMat(photo), 512); val mask = segment(m, seedX, seedY)
        val out = Bitmap.createBitmap(mask.cols(), mask.rows(), Bitmap.Config.ARGB_8888)
        val vis = Mat(); Core.multiply(mask, Scalar(255.0), vis); val rgba = Mat(); Imgproc.cvtColor(vis, rgba, Imgproc.COLOR_GRAY2RGBA)
        Utils.matToBitmap(rgba, out); return out
    }

    /** Noise floor from ≥ 3 same-day captures of the unchanged spot: std of area ratio and contrast vs the first. */
    fun noiseFloor(captures: List<Bitmap>, seedX: Double, seedY: Double): NoiseFloor? {
        if (captures.size < 3) return null
        val ms = captures.drop(1).map { compute(captures[0], it, seedX, seedY, null, null, null, null, true) }.filter { it.alignOk }
        if (ms.size < 2) return null
        fun sd(x: List<Double>): Double { val m = x.average(); return sqrt(x.sumOf { (it - m).pow(2) } / x.size) }
        val areas = ms.mapNotNull { it.areaRatio?.minus(1.0) }; val cons = ms.mapNotNull { it.contrastDelta }
        return NoiseFloor(maxOf(sd(areas), areas.maxOfOrNull { kotlin.math.abs(it) } ?: 0.0), maxOf(sd(cons), cons.maxOfOrNull { kotlin.math.abs(it) } ?: 0.0), captures.size)
    }
}
