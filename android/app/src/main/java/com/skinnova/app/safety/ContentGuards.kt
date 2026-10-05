package com.skinnova.app.safety

/** contracts §3 rule 5. Python twin: ml/llm/validate.py guard_text. rx terms from assets/safety/rx_terms.txt. */
class ContentGuards(private val rxTerms: List<String>) {
    companion object {
        val DOSE = Regex("""\b\d+(\.\d+)?\s?(mg|mcg|g|ml|%|IU)(?![A-Za-z])""", RegexOption.IGNORE_CASE)
        val DIAGNOSIS = Regex("""\b(you have|you are diagnosed|this is definitely)\b""", RegexOption.IGNORE_CASE)

        fun parseTerms(text: String) = text.lines().map { it.trim().lowercase() }.filter { it.isNotEmpty() && !it.startsWith("#") }
    }

    private val rxRegex = rxTerms.map { it to Regex("""\b${Regex.escape(it)}\b""") }

    fun check(text: String): List<String> {
        val errs = mutableListOf<String>()
        DOSE.find(text)?.let { errs += "dose_pattern:${it.value}" }
        val low = text.lowercase()
        for ((term, rx) in rxRegex) if (rx.containsMatchIn(low)) errs += "rx_term:$term"
        DIAGNOSIS.find(text)?.let { errs += "diagnosis_phrasing:${it.value.lowercase()}" }
        return errs
    }
}
