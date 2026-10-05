package com.skinnova.app.report

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.Typeface
import android.graphics.pdf.PdfDocument
import com.skinnova.app.BuildConfig
import com.skinnova.app.R
import com.skinnova.app.data.SpotEntity
import com.skinnova.app.timeline.ChangeMetrics
import java.io.File
import java.text.DateFormat
import java.util.Date

/** Doctor Visit Summary (contracts §11): A4 PDF made on the phone; shared only on the user's tap (FileProvider). */
object DoctorSummaryPdf {
    data class Row(val date: Long, val photo: Bitmap?, val cvTop3: String, val tier: String, val metrics: ChangeMetrics?, val narration: String?)

    fun build(ctx: Context, spot: SpotEntity, rows: List<Row>, latestAnswers: String, ruleMessages: List<String>, questions: List<String>, modelSha: String): File {
        val doc = PdfDocument()
        val W = 595; val H = 842; val M = 40
        val title = Paint().apply { textSize = 18f; typeface = Typeface.DEFAULT_BOLD }
        val body = Paint().apply { textSize = 10f; color = Color.DKGRAY }
        val small = Paint().apply { textSize = 8f; color = Color.GRAY }
        var page = doc.startPage(PdfDocument.PageInfo.Builder(W, H, 1).create())
        var cv: Canvas = page.canvas
        var y = M + 10f
        fun line(t: String, p: Paint = body, dy: Float = 14f) { wrap(t, p, W - 2 * M).forEach { cv.drawText(it, M.toFloat(), y, p); y += dy } }
        fun newPage(n: Int) { doc.finishPage(page); page = doc.startPage(PdfDocument.PageInfo.Builder(W, H, n).create()); cv = page.canvas; y = M + 10f }
        val df = DateFormat.getDateInstance(DateFormat.MEDIUM)
        line("SkinNova — Doctor Visit Summary", title, 24f)
        line("Generated ${df.format(Date())} · app ${BuildConfig.VERSION_NAME} · model ${modelSha.take(12).ifEmpty { "basic mode" }}", small, 12f)
        line(ctx.getString(R.string.disclaimer), small, 12f); y += 8
        line("Spot: ${spot.name} · body site: ${spot.bodySite} · tracked since ${df.format(Date(spot.createdAt))}", body)
        line("Coin reference: ${spot.coinDiameterMm?.let { "%.1f mm".format(it) } ?: "none"} · re-check every ${spot.reminderDays} days", body); y += 6
        line("Captures", title.apply { textSize = 13f }, 18f)
        var n = 1
        for (r in rows) {
            if (y > H - 170) newPage(++n)
            r.photo?.let { cv.drawBitmap(it, null, Rect(M, y.toInt(), M + 110, y.toInt() + 110), null) }
            val x = M + 122f; var yy = y + 10
            fun t(s: String) { wrap(s, body, W - M - x.toInt()).forEach { cv.drawText(it, x, yy, body); yy += 13 } }
            t(df.format(Date(r.date)) + " · tier ${r.tier}")
            t("Image model top-3: ${r.cvTop3}")
            r.metrics?.let { m -> t("Area ratio ${m.areaRatio?.let { "%.2f".format(it) } ?: "—"} · contrast Δ ${m.contrastDelta?.let { "%.1f".format(it) } ?: "—"} · confidence ${m.confidence}") }
            r.narration?.let { t(it) }
            y = maxOf(yy, y + 118) + 8
        }
        if (y > H - 200) newPage(++n)
        line("Latest answers", title, 18f); line(latestAnswers)
        if (ruleMessages.isNotEmpty()) { y += 6; line("Safety messages shown", title, 18f); ruleMessages.forEach { line("• $it") } }
        if (questions.isNotEmpty()) { y += 6; line("Questions to ask your doctor", title, 18f); questions.forEach { line("• $it") } }
        y += 10; line(ctx.getString(R.string.disclaimer), small, 12f)
        doc.finishPage(page)
        val dir = File(ctx.cacheDir, "reports").apply { mkdirs() }
        val f = File(dir, "skinnova_${spot.name.replace(Regex("[^A-Za-z0-9]"), "_")}.pdf")
        f.outputStream().use { doc.writeTo(it) }; doc.close()
        return f
    }

    private fun wrap(t: String, p: Paint, width: Int): List<String> {
        val out = ArrayList<String>(); var cur = ""
        for (w in t.split(" ")) { val cand = if (cur.isEmpty()) w else "$cur $w"; if (p.measureText(cand) > width && cur.isNotEmpty()) { out += cur; cur = w } else cur = cand }
        if (cur.isNotEmpty()) out += cur; return out
    }
}
