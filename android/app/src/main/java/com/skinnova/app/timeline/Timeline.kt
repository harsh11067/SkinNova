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
import org.opencv.features2d.SIFT
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
    @SerialName("seg_ok") val segOk: Boolean = true,
    @SerialName("coin_scale_err") val coinScaleErr: Double? = null,
    @SerialName("area_mm2_base") val areaMm2Base: Double? = null,
    @SerialName("area_raw") val areaRaw: Double? = null,
    @SerialName("contrast_raw") val contrastRaw: Double? = null,
) {
    fun json() = SnJson.encodeToString(serializer(), this)
}

object Timeline {
    const val ALIGN_MIN_RATIO = 0.25
    const val ALIGN_MIN_INLIERS = 30
    /** Coin disc × this (rim + contact shadow) is background for the lesion segmentation (metrics.py COIN_EXCLUDE). */
    const val COIN_EXCLUDE = 1.15
    /** Loaded in the object initializer: runs before ANY Timeline method (camera-frame callbacks included). */
    val ready: Boolean = OpenCVLoader.initLocal()

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

    /** H maps new → base. CLAHE grey + SIFT (v1 ORB aligned only 65 % of synthetic same-spot pairs on smooth skin). */
    fun align(baseRgb: Mat, newRgb: Mat): Alignment {
        Core.setRNGSeed(0)
        val g1 = Mat(); val g2 = Mat()
        Imgproc.cvtColor(baseRgb, g1, Imgproc.COLOR_RGB2GRAY); Imgproc.cvtColor(newRgb, g2, Imgproc.COLOR_RGB2GRAY)
        val clahe = Imgproc.createCLAHE(2.0, Size(8.0, 8.0)); clahe.apply(g1, g1); clahe.apply(g2, g2)
        val sift = SIFT.create(2000)
        val k1 = MatOfKeyPoint(); val k2 = MatOfKeyPoint(); val d1 = Mat(); val d2 = Mat()
        sift.detectAndCompute(g1, Mat(), k1, d1); sift.detectAndCompute(g2, Mat(), k2, d2)
        val kp1 = k1.toArray(); val kp2 = k2.toArray()
        if (d1.empty() || d2.empty() || kp1.size < 10 || kp2.size < 10) return Alignment(null, 0.0, 0)
        val knn = ArrayList<MatOfDMatch>()
        BFMatcher.create(Core.NORM_L2, false).knnMatch(d2, d1, knn, 2)
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

    /** GrabCut seeded at the tap (normalised x,y). Returns CV_8U 0/1 mask. [exclude] = (cx, cy, r) of a detected coin → background. */
    fun segment(rgb: Mat, seedX: Double, seedY: Double, rFrac: Double = 0.12, iters: Int = 5, exclude: DoubleArray? = null): Mat {
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
        if (exclude != null) Imgproc.circle(mask, Point(exclude[0].toInt().toDouble(), exclude[1].toInt().toDouble()), (COIN_EXCLUDE * exclude[2]).toInt(),
            Scalar(Imgproc.GC_BGD.toDouble()), -1)
        otsuInit(rgb, mask, sx, sy, r)   // metrics.py SEG_INIT = "otsu"
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

    /**
     * metrics.py _otsu_init: inside the probable-lesion disc (1.6 r), probable FG where the Lab distance from the surrounding
     * skin (mean of the annulus 1.6–2.2 r, frame/coin excluded) is above Otsu's threshold, probable BG elsewhere.
     * 8-bit OpenCV Lab, double maths, half-even rounding (Math.rint ≡ numpy rint) — bit-identical to Python.
     */
    private fun otsuInit(rgb: Mat, mask: Mat, sx: Int, sy: Int, r: Double) {
        val h = mask.rows(); val w = mask.cols(); val c = Point(sx.toDouble(), sy.toDouble()); val n = w * h
        val disc = Mat.zeros(h, w, CvType.CV_8U); Imgproc.circle(disc, c, (1.6 * r).toInt(), Scalar(1.0), -1)
        val ann = Mat.zeros(h, w, CvType.CV_8U); Imgproc.circle(ann, c, (2.2 * r).toInt(), Scalar(1.0), -1)
        val mk = ByteArray(n); mask.get(0, 0, mk); val dk = ByteArray(n); disc.get(0, 0, dk); val ak = ByteArray(n); ann.get(0, 0, ak)
        val lab = Mat(); Imgproc.cvtColor(rgb, lab, Imgproc.COLOR_RGB2Lab); val lk = ByteArray(n * 3); lab.get(0, 0, lk)
        var cnt = 0; var sL = 0.0; var sA = 0.0; var sB = 0.0
        for (i in 0 until n) if (ak[i].toInt() != 0 && dk[i].toInt() == 0 && mk[i].toInt() != Imgproc.GC_BGD) {
            cnt++; sL += lk[3 * i].toInt() and 255; sA += lk[3 * i + 1].toInt() and 255; sB += lk[3 * i + 2].toInt() and 255
        }
        if (cnt < 50) return
        val mL = sL / cnt; val mA = sA / cnt; val mB = sB / cnt
        val pr = ArrayList<Int>(); val d8 = IntArray(n)
        for (i in 0 until n) {
            val m = mk[i].toInt()
            if (dk[i].toInt() == 0 || m == Imgproc.GC_BGD || m == Imgproc.GC_FGD) continue
            val dl = (lk[3 * i].toInt() and 255) - mL; val da = (lk[3 * i + 1].toInt() and 255) - mA; val db = (lk[3 * i + 2].toInt() and 255) - mB
            d8[i] = Math.rint(2.0 * sqrt(dl * dl + da * da + db * db)).coerceIn(0.0, 255.0).toInt(); pr += i
        }
        if (pr.size < 50) return
        val col = Mat(pr.size, 1, CvType.CV_8U); col.put(0, 0, ByteArray(pr.size) { d8[pr[it]].toByte() })
        val t = Imgproc.threshold(col, Mat(), 0.0, 255.0, Imgproc.THRESH_BINARY or Imgproc.THRESH_OTSU)
        for (i in pr) mk[i] = (if (d8[i] > t) Imgproc.GC_PR_FGD else Imgproc.GC_PR_BGD).toByte()
        mask.put(0, 0, mk)
    }

    /** Mask with prior r, and its IoU against the mask with prior 1.4 r (unstable segmentation → confidence low). */
    fun segmentStable(rgb: Mat, seedX: Double, seedY: Double, rFrac: Double, exclude: DoubleArray? = null): Pair<Mat, Double> {
        val a = segment(rgb, seedX, seedY, rFrac, exclude = exclude); val b = segment(rgb, seedX, seedY, 1.4 * rFrac, exclude = exclude)
        val inter = Mat(); Core.bitwise_and(a, b, inter); val uni = Mat(); Core.bitwise_or(a, b, uni)
        val u = Core.countNonZero(uni)
        return a to (if (u == 0) 0.0 else Core.countNonZero(inter).toDouble() / u)
    }

    /**
     * (cx, cy, r) of a coin, or null. HOUGH_GRADIENT_ALT candidates (radius 4–15 % of the short side) verified as metal;
     * a circle containing the tapped spot [seedPx] (this photo's pixels) is the lesion, never the coin.
     */
    fun detectCoin(rgb: Mat, seedPx: DoubleArray? = null): DoubleArray? {
        val g = Mat(); Imgproc.cvtColor(rgb, g, Imgproc.COLOR_RGB2GRAY); Imgproc.medianBlur(g, g, 3)
        val m = min(g.cols(), g.rows())
        val c = Mat()
        Imgproc.HoughCircles(g, c, Imgproc.HOUGH_GRADIENT_ALT, 1.5, m * 0.05, 300.0, 0.8, (0.04 * m).toInt(), (0.15 * m).toInt())
        val hsv = Mat(); Imgproc.cvtColor(rgb, hsv, Imgproc.COLOR_RGB2HSV)
        val sat = Mat(); Core.extractChannel(hsv, sat, 1)
        val edges = Mat(); Imgproc.Canny(g, edges, 30.0, 90.0)
        for (i in 0 until c.cols()) {
            val v = c.get(0, i)
            if (seedPx != null && kotlin.math.hypot(v[0] - seedPx[0], v[1] - seedPx[1]) <= v[2]) continue   // that is the lesion
            val disc = Mat.zeros(g.size(), CvType.CV_8U)
            Imgproc.circle(disc, Point(v[0], v[1]), (v[2] * 0.85).toInt(), Scalar(1.0), -1)
            if (Core.countNonZero(disc) == 0 || Core.mean(sat, disc).`val`[0] / 255.0 > 0.35) continue   // coloured blob, not metal
            var support = 0
            for (k in 0 until 72) {
                val t = 2 * PI * k / 72; var ok = false
                for (dr in -2..2) {
                    val px = Math.round(v[0] + (v[2] + dr) * kotlin.math.cos(t)).toInt(); val py = Math.round(v[1] + (v[2] + dr) * kotlin.math.sin(t)).toInt()
                    if (px in 0 until g.cols() && py in 0 until g.rows() && edges.get(py, px)[0] > 0) { ok = true; break }
                }
                if (ok) support++
            }
            if (support / 72.0 < 0.5) continue
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

    /**
     * ml/timeline/metrics.py skin_normalised_lab (SKIN_NORM = "mean") + lesion_contrast: the new photo is re-lit onto the
     * base photo by matching the surrounding-skin ring's mean Lab colour (exposure / white-balance offsets cancel; a real
     * change of the lesion relative to its own skin remains). Chosen on TL1: mean-only beat no normalisation and mean+spread
     * on both gates (light-only drift 0.92 ΔE, colour-change error 1.53 ΔE). lesion' = lesion − μ_new + μ_base, ring' = μ_base.
     */
    fun lesionContrastRelit(newRgb: Mat, m1: Mat, baseRgb: Mat, m0: Mat): Double {
        val lesion = meanLab(newRgb, m1) ?: return 0.0
        val muN = ringMean(newRgb, m1); val muB = ringMean(baseRgb, m0)
        if (muN == null || muB == null) return lesionContrast(newRgb, m1)
        return Colour.ciede2000(DoubleArray(3) { lesion[it] - muN[it] + muB[it] }, muB)
    }

    /** Mean Lab of the skin ring around [mask]; null below 20 pixels (same threshold as Python). */
    private fun ringMean(rgb: Mat, mask: Mat): DoubleArray? {
        val r = ring(mask)
        return if (Core.countNonZero(r) < 20) null else meanLab(rgb, r)
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
        // coins first (each in its own photo, the tap mapped into the new photo by H⁻¹) so their discs are background for
        // GrabCut; adaptive prior + stability (ml/timeline/metrics.py): area ratio from homography-aligned masks
        // (scale-free); the coin is a scale CHECK, not the area source
        val h = al.h!!
        val homScale = sqrt(kotlin.math.abs(h.get(0, 0)[0] * h.get(1, 1)[0] - h.get(0, 1)[0] * h.get(1, 0)[0]))
        val c0 = detectCoin(b, doubleArrayOf(seedX * b.cols(), seedY * b.rows())); val c1 = detectCoin(n, seedInNew(h, seedX, seedY, b))
        val c1w = c1?.let { val q = project(h, it[0], it[1]); doubleArrayOf(q[0], q[1], it[2] * homScale) }   // in the base frame
        val mInit = segment(b, seedX, seedY, exclude = c0)
        val rFrac = maxOf(0.12, 1.25 * sqrt(maxOf(1, Core.countNonZero(mInit)).toDouble() / PI) / min(b.cols(), b.rows()))
        val (m0, iou0) = segmentStable(b, seedX, seedY, rFrac, c0); val (m1, iou1) = segmentStable(warped, seedX, seedY, rFrac, c1w)
        val segOk = minOf(iou0, iou1) >= 0.75
        val a0 = Core.countNonZero(m0).toDouble(); val a1 = Core.countNonZero(m1).toDouble()
        val areaRatio = if (a0 > 0) a1 / a0 else null
        val coinBoth = c0 != null && c1 != null && coinMm != null
        val coinScaleErr = if (coinBoth) kotlin.math.abs((c0!![2] / c1!![2]) / homScale - 1) else null
        val coinOk = coinBoth && coinScaleErr!! <= 0.10
        val cd = lesionContrastRelit(warped, m1, b, m0) - lesionContrast(b, m0)
        val conf = if (coinOk && segOk && (noise?.n ?: 0) >= 3) "ok" else "low"
        return ChangeMetrics(alignScore = al.ratio, alignInliers = al.inliers, alignOk = true, coinInBoth = coinBoth, areaRatio = areaRatio,
            contrastDelta = cd, borderIrregularityDelta = irregularity(m1) - irregularity(m0),
            cvShiftJs = if (cvBase != null && cvNew != null) Colour.jsDivergence(cvBase, cvNew) else null,
            noiseFloor = noise, confidence = conf, lesionType = lesionType, segOk = segOk, coinScaleErr = coinScaleErr,
            areaMm2Base = if (coinBoth) a0 * (coinMm!! / (2 * c0!![2])).pow(2) else null)
    }

    /** p' = H·p (homogeneous). */
    fun project(h: Mat, x: Double, y: Double): DoubleArray {
        val w = h.get(2, 0)[0] * x + h.get(2, 1)[0] * y + h.get(2, 2)[0]
        return doubleArrayOf((h.get(0, 0)[0] * x + h.get(0, 1)[0] * y + h.get(0, 2)[0]) / w, (h.get(1, 0)[0] * x + h.get(1, 1)[0] * y + h.get(1, 2)[0]) / w)
    }

    /** The tap (normalised in [base]) in the new photo's pixels: H maps new → base, so H⁻¹. */
    fun seedInNew(h: Mat, seedX: Double, seedY: Double, base: Mat): DoubleArray =
        project(h.inv(), seedX * base.cols(), seedY * base.rows())

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
