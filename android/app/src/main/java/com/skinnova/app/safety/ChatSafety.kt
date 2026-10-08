package com.skinnova.app.safety

/**
 * Ask SkinNova (follow-up questions): deterministic checks around the LLM answer.
 * - [dangerSigns]: the question mentions a danger sign → the app shows a fixed "get care today" banner (strings.xml),
 *   whatever the model writes. It never lowers the result's advice level.
 * - [clean]: an answer that names a prescription medicine, a dose or a diagnosis ("you have …") is replaced by a safe
 *   message; a JSON-shaped reply (the model's analysis habit) is not shown either.
 */
object ChatSafety {
    private val DANGER = listOf(
        Regex("""\bfever|\bhigh temperature|\bspreading (fast|quickly|rapidly)|\bspreads? (fast|quickly)""", RegexOption.IGNORE_CASE),
        Regex("""\b(can'?t|cannot|hard to|trouble|difficulty) breath""", RegexOption.IGNORE_CASE),
        Regex("""\bswollen (lips?|face|tongue|eyes?|throat)|\b(lips?|face|tongue|throat|eyes?) (is |are )?(swollen|swelling)""", RegexOption.IGNORE_CASE),
        Regex("""\b(in|near|around) (my |the )?eyes?\b|\beye (pain|swelling)|\bmouth sores?""", RegexOption.IGNORE_CASE),
        Regex("""\bbleed(ing|s)? (won'?t|doesn'?t|does not|will not) stop|\bchest pain|\bfaint(ed|ing)?\b|\bunconscious|\bpus\b""", RegexOption.IGNORE_CASE),
        Regex("""बुखार|साँस|सांस|आँख|आंख|तेज़ी से फैल|तेजी से फैल|खून नहीं रुक|सूजन|मवाद|बेहोश"""),
    )

    fun dangerSigns(question: String): Boolean = DANGER.any { it.containsMatchIn(question) }

    /** A diagnosis statement. Narrower than the analysis validator's "you have": answers say "if you have a fever …". */
    private val DIAGNOSIS = Regex(
        """\byou (definitely |probably |clearly )?have (a |an )?(eczema|atopic|dermatitis|psoriasis|acne|vitiligo|scabies|ringworm|tinea|""" +
            """fungal infection|skin cancer|cancer|melanoma|carcinoma)\b|\byou are diagnosed|\bthis is definitely""", RegexOption.IGNORE_CASE)

    /** Sentences the small model invents (found by ml/eval/chat_probe.py): places/distances, and timeline advice
     *  ("retake with a coin") nobody asked for. Removed sentence by sentence; Python twin: chat_probe.tidy. */
    private val UNSUPPORTED = Regex(
        """\b\d+([.,]\d+)?\s*(miles?|km|kilomet\w*)\b|\b(care|health|medical) cent(re|er)s?\b|\bcoin\b|\bsikka\b|सिक्का""", RegexOption.IGNORE_CASE)
    private val SENTENCE = Regex("""(?<=[.!?।])\s+""")

    fun tidy(answer: String): String = SENTENCE.split(answer.trim()).filterNot { UNSUPPORTED.containsMatchIn(it) }.joinToString(" ").trim()

    /** null = the answer may be shown; otherwise why it was withheld. */
    fun withheld(answer: String, guards: ContentGuards): String? = when {
        answer.isBlank() -> "empty"
        answer.trimStart().startsWith("{") || answer.trimStart().startsWith("[") -> "json"
        DIAGNOSIS.containsMatchIn(answer) -> "diagnosis"
        else -> guards.check(answer).firstOrNull { !it.startsWith("diagnosis_phrasing") }
    }
}
