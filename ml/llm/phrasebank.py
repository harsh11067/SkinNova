"""Phrase banks for SFT tasks T6 (intake extraction) and T8 (localization).

T6: field value → phrasings in three styles: en (English), hl (Hinglish, Roman script), hd (Hindi, Devanagari).
    The LAST phrasing of every list is held out of training and used only for the eval split, so extraction
    accuracy is measured on wording the model never saw.
T8: English ↔ Hindi sentence pairs. Hindi written for this project; must be reviewed by a native speaker (d2y.md).
"""
from __future__ import annotations

INTAKE = {
    "body_site": {
        "face": {"en": ["on my face", "on my cheek", "on my forehead", "near my nose"],
                 "hl": ["chehre par", "gaal par", "maathe par", "naak ke paas"], "hd": ["चेहरे पर", "गाल पर", "माथे पर"]},
        "scalp": {"en": ["on my scalp", "in my hair", "on my head under the hair"], "hl": ["sir mein", "baalon mein", "sar par baalon ke neeche"],
                  "hd": ["सिर में", "बालों में"]},
        "neck": {"en": ["on my neck", "on the side of my neck"], "hl": ["gardan par", "gale par"], "hd": ["गर्दन पर", "गले पर"]},
        "chest": {"en": ["on my chest", "on my upper chest"], "hl": ["chhati par", "seene par"], "hd": ["छाती पर", "सीने पर"]},
        "back": {"en": ["on my back", "on my upper back", "on my lower back"], "hl": ["peeth par", "kamar par", "peeth ke upar"],
                 "hd": ["पीठ पर", "कमर पर"]},
        "abdomen": {"en": ["on my stomach", "on my belly", "around my navel"], "hl": ["pet par", "naabhi ke paas"], "hd": ["पेट पर", "नाभि के पास"]},
        "arm": {"en": ["on my arm", "on my forearm", "on my elbow"], "hl": ["baanh par", "kohni par", "haath ke upar wale hisse par"],
                "hd": ["बाँह पर", "कोहनी पर"]},
        "hand": {"en": ["on my hand", "on my fingers", "on my palm", "on the back of my hand"],
                 "hl": ["haath par", "ungliyon mein", "hatheli par"], "hd": ["हाथ पर", "उंगलियों में", "हथेली पर"]},
        "leg": {"en": ["on my leg", "on my thigh", "on my knee", "on my shin"], "hl": ["pair par", "jaangh par", "ghutne par", "taang par"],
                "hd": ["पैर पर", "जांघ पर", "घुटने पर"]},
        "foot": {"en": ["on my foot", "between my toes", "on the sole of my foot"], "hl": ["pair ke panje par", "pair ki ungliyon ke beech", "talve par"],
                 "hd": ["तलवे पर", "पैर की उंगलियों के बीच"]},
        "groin": {"en": ["in my groin", "on my inner thigh near the groin"], "hl": ["jaangh ke jod par", "private part ke paas"],
                  "hd": ["जांघ के जोड़ पर", "जांघ के अंदर की तरफ़"]},
        "nails": {"en": ["on my toenail", "on my fingernail"], "hl": ["naakhun mein", "pair ke naakhun mein"], "hd": ["नाखून में", "पैर के नाखून में"]},
    },
    "duration": {
        "lt_1w": {"en": ["since two days", "for three days", "since yesterday", "for four days"], "hl": ["do din se", "kal se", "teen din se", "chaar din se"],
                  "hd": ["दो दिन से", "कल से", "तीन दिन से"]},
        "1_4w": {"en": ["for two weeks", "for three weeks", "for about ten days", "for 2 weeks"], "hl": ["do hafte se", "teen hafte se", "das din se", "2 hafte se"],
                 "hd": ["दो हफ़्ते से", "तीन हफ़्ते से", "दस दिन से"]},
        "1_6m": {"en": ["for two months", "for three months", "for four months", "for 5 months"], "hl": ["do mahine se", "teen mahine se", "chaar mahine se", "5 mahine se"],
                 "hd": ["दो महीने से", "तीन महीने से", "चार महीने से"]},
        "gt_6m": {"en": ["for over a year", "for two years", "since childhood", "for many years"], "hl": ["ek saal se", "do saal se", "bachpan se", "kai saalon se"],
                  "hd": ["एक साल से", "बचपन से", "कई सालों से"]},
    },
    "itch": {
        0: {"en": ["it does not itch", "there is no itching at all"], "hl": ["khujli nahi hoti", "bilkul khujli nahi hai"], "hd": ["खुजली नहीं होती", "बिल्कुल खुजली नहीं है"]},
        1: {"en": ["it itches a little", "there is mild itching"], "hl": ["thodi khujli hoti hai", "halki si khujli hai"], "hd": ["थोड़ी खुजली होती है", "हल्की सी खुजली है"]},
        2: {"en": ["it itches quite a lot", "the itching bothers me during the day"], "hl": ["kaafi khujli hoti hai", "din mein bhi khujli hoti hai"],
            "hd": ["काफ़ी खुजली होती है", "दिन में भी खुजली होती है"]},
        3: {"en": ["it itches badly at night", "the itching is unbearable", "the itching keeps me awake"],
            "hl": ["bahut khujli hoti hai", "raat ko bahut khujli hoti hai", "khujli se neend nahi aati"], "hd": ["बहुत खुजली होती है", "रात को बहुत खुजली होती है"]},
    },
    "pain": {
        0: {"en": ["it does not hurt", "there is no pain"], "hl": ["dard nahi hai", "bilkul dard nahi hota"], "hd": ["दर्द नहीं है", "बिल्कुल दर्द नहीं होता"]},
        1: {"en": ["it hurts a little", "there is slight pain"], "hl": ["thoda dard hai", "halka dard hota hai"], "hd": ["थोड़ा दर्द है", "हल्का दर्द होता है"]},
        2: {"en": ["it is quite painful", "it hurts when I touch it"], "hl": ["kaafi dard hota hai", "chhoone par dard hota hai"], "hd": ["काफ़ी दर्द होता है", "छूने पर दर्द होता है"]},
        3: {"en": ["the pain is severe", "it hurts a lot"], "hl": ["bahut dard hota hai", "tez dard hai"], "hd": ["बहुत दर्द होता है", "तेज़ दर्द है"]},
    },
    "changing": {
        "no": {"en": ["it has not changed", "it looks the same as before"], "hl": ["waisa hi hai", "bilkul badla nahi hai"], "hd": ["वैसा ही है", "बिल्कुल बदला नहीं है"]},
        "growing": {"en": ["it is getting bigger", "it is growing"], "hl": ["bada ho raha hai", "badhta ja raha hai"], "hd": ["बड़ा हो रहा है", "बढ़ता जा रहा है"]},
        "changing_color": {"en": ["it is getting darker", "its colour is changing"], "hl": ["rang badal raha hai", "kaala hota ja raha hai"], "hd": ["रंग बदल रहा है", "काला होता जा रहा है"]},
        "changing_shape": {"en": ["its shape is changing", "the edge looks uneven now"], "hl": ["shape badal rahi hai", "aakaar badal raha hai"], "hd": ["आकार बदल रहा है", "किनारा टेढ़ा हो गया है"]},
        "spreading": {"en": ["it is spreading", "it has spread to other places"], "hl": ["phail raha hai", "aur jagah bhi phail gaya hai"], "hd": ["फैल रहा है", "और जगह भी फैल गया है"]},
        "unsure": {"en": ["I am not sure if it changed", "I can't tell if it changed"], "hl": ["pata nahi badla ya nahi", "samajh nahi aata badla ya nahi"],
                   "hd": ["पता नहीं बदला या नहीं", "समझ नहीं आता बदला या नहीं"]},
    },
    "bleeding_or_crusting": {
        True: {"en": ["it bleeds sometimes", "there is crusting on it", "it oozes fluid"], "hl": ["kabhi kabhi khoon aata hai", "papdi jam jaati hai", "paani nikalta hai"],
               "hd": ["कभी-कभी खून आता है", "पपड़ी जम जाती है"]},
        False: {"en": ["it does not bleed", "there is no bleeding"], "hl": ["khoon nahi aata", "koi khoon nahi nikalta"], "hd": ["खून नहीं आता", "कोई खून नहीं निकलता"]},
    },
    "fever_or_unwell": {
        True: {"en": ["I have a fever too", "I feel unwell", "I also have fever"], "hl": ["bukhar bhi hai", "tabiyat kharab hai", "bukhar aa raha hai"],
               "hd": ["बुखार भी है", "तबीयत खराब है"]},
        False: {"en": ["there is no fever", "otherwise I feel fine"], "hl": ["bukhar nahi hai", "baaki tabiyat theek hai"], "hd": ["बुखार नहीं है", "बाकी तबीयत ठीक है"]},
    },
    "others_affected": {
        True: {"en": ["my brother has it too", "others at home have the same thing", "my child also has it"],
               "hl": ["bhai ko bhi hai", "ghar mein sabko ho raha hai", "bachche ko bhi hai"], "hd": ["भाई को भी है", "घर में सबको हो रहा है"]},
        False: {"en": ["nobody else at home has it", "no one else has it"], "hl": ["ghar mein kisi aur ko nahi hai", "aur kisi ko nahi hai"],
                "hd": ["घर में किसी और को नहीं है", "और किसी को नहीं है"]},
    },
    "new_product_or_exposure": {
        True: {"en": ["it started after I used a new soap", "it began after a new cream", "it started after I wore a new watch"],
               "hl": ["naya sabun lagane ke baad shuru hua", "nayi cream lagane ke baad hua", "nayi ghadi pehenne ke baad shuru hua"],
               "hd": ["नया साबुन लगाने के बाद शुरू हुआ", "नई क्रीम लगाने के बाद हुआ"]},
        False: {"en": ["I have not changed any product", "I did not use anything new"], "hl": ["koi naya product use nahi kiya", "kuch naya nahi lagaya"],
                "hd": ["कोई नया प्रोडक्ट इस्तेमाल नहीं किया", "कुछ नया नहीं लगाया"]},
    },
    "age_band": {
        "lt_12": {"en": ["my 8 year old son", "my daughter is 6"], "hl": ["mere 8 saal ke bete ko", "meri 6 saal ki beti ko"], "hd": ["मेरे 8 साल के बेटे को", "मेरी 6 साल की बेटी को"]},
        "12_17": {"en": ["I am 15", "I am sixteen years old"], "hl": ["meri umar 15 saal hai", "main 16 saal ka hoon"], "hd": ["मेरी उम्र 15 साल है", "मैं 16 साल का हूँ"]},
        "18_39": {"en": ["I am 28 years old", "I am 33"], "hl": ["main 28 saal ka hoon", "meri umar 30 saal hai"], "hd": ["मेरी उम्र 30 साल है", "मैं 28 साल का हूँ"]},
        "40_59": {"en": ["I am 45", "I am 52 years old"], "hl": ["main 50 saal ki hoon", "meri umar 45 saal hai"], "hd": ["मेरी उम्र 45 साल है", "मैं 50 साल की हूँ"]},
        "60_plus": {"en": ["I am 67 years old", "my mother is 72"], "hl": ["meri umar 70 saal hai", "meri maa 72 saal ki hain"], "hd": ["मेरी उम्र 70 साल है", "मेरी माँ 72 साल की हैं"]},
    },
}

FILLERS = {"en": ["Doctor,", "Hello,", "So,", "Basically,", ""], "hl": ["Doctor sahab,", "Namaste,", "Dekhiye,", "Basically", ""],
           "hd": ["डॉक्टर साहब,", "नमस्ते,", "देखिए,", ""]}
NOTES = {"en": [("there is a round patch", "round patch"), ("it looks like a ring", "ring-shaped")],
         "hl": [("gol daag hai", "gol daag (round patch)"), ("chhote chhote daane hain", "chhote daane (small bumps)")],
         "hd": [("गोल दाग है", "गोल दाग (round patch)"), ("छोटे-छोटे दाने हैं", "छोटे दाने (small bumps)")]}
LANG_CODE = {"en": "en", "hl": "hi", "hd": "hi"}

# ---------------- T8 bilingual pieces ----------------
SUMMARY_HI = {
    "eczema_atopic": "यह सूखी, खुजली वाली और सूजी हुई त्वचा की लंबी चलने वाली प्रवृत्ति है जो बार-बार उभरती है। यह छूत की बीमारी नहीं है।",
    "contact_dermatitis": "यह त्वचा पर किसी चीज़ के छूने से होने वाला दाना है, जैसे साबुन, डिटर्जेंट, निकल या खुशबू।",
    "seborrheic_dermatitis": "यह तैलीय हिस्सों में होने वाला आम, हानिरहित पपड़ीदार दाना है जो आता-जाता रहता है।",
    "tinea": "यह त्वचा का फंगल संक्रमण (दाद) है। यह छूने, तौलिये और कपड़ों से फैलता है।",
    "scabies": "यह बहुत खुजली वाला दाना है जो त्वचा में घुसने वाले बहुत छोटे कीड़ों से होता है। यह नज़दीकी संपर्क से फैलता है।",
    "acne": "यह तेल और सूजन से बंद रोमछिद्रों के कारण होने वाले दाने हैं।",
    "psoriasis": "यह लंबे समय तक चलने वाली प्रतिरक्षा से जुड़ी स्थिति है जिसमें त्वचा पर मोटे, पपड़ीदार चकत्ते बनते हैं। यह छूत की बीमारी नहीं है।",
    "vitiligo": "इसमें त्वचा के कुछ हिस्सों का रंग चला जाता है। यह छूत की बीमारी नहीं है और इसमें दर्द नहीं होता।",
    "benign_lesion": "ये आम हानिरहित उभार हैं, जैसे तिल। ये आमतौर पर सालों तक एक जैसे रहते हैं।",
    "suspicious_lesion": "यह ऐसा दाग है जिसे डॉक्टर खुद देखना चाहेंगे। ज़्यादातर जाँचे गए दाग हानिरहित निकलते हैं, लेकिन जल्दी जाँच ज़रूरी है।",
    "other": "फ़ोटो SkinNova की जानी-पहचानी श्रेणियों से साफ़ तौर पर मेल नहीं खाती।",
}
CARE_HI = {
    "moisturise often with a plain, fragrance-free cream": "बिना खुशबू वाली सादी क्रीम से बार-बार नमी दें",
    "short lukewarm baths and gentle soap-free wash": "थोड़ी देर गुनगुने पानी से नहाएँ और बिना साबुन वाला हल्का क्लींज़र इस्तेमाल करें",
    "keep nails short and avoid known triggers": "नाखून छोटे रखें और जानी हुई वजहों से बचें",
    "stop using the suspected product": "जिस प्रोडक्ट पर शक है उसका इस्तेमाल बंद करें",
    "rinse the area with plain cool water": "उस हिस्से को सादे ठंडे पानी से धोएँ",
    "wear gloves for wet or chemical work": "गीले या केमिकल वाले काम के लिए दस्ताने पहनें",
    "use plain unscented moisturiser": "बिना खुशबू वाली सादी मॉइस्चराइज़र लगाएँ",
    "wash the scalp regularly with a gentle shampoo": "हल्के शैम्पू से सिर नियमित रूप से धोएँ",
    "avoid harsh scrubbing": "ज़ोर से रगड़ने से बचें",
    "manage stress and sleep, which can trigger flares": "तनाव और नींद का ध्यान रखें, इनसे समस्या बढ़ सकती है",
    "keep the area clean and fully dry": "उस हिस्से को साफ़ और पूरी तरह सूखा रखें",
    "wear loose cotton clothes and change daily": "ढीले सूती कपड़े पहनें और रोज़ बदलें",
    "do not share towels, combs or clothes": "तौलिया, कंघी या कपड़े किसी के साथ न बाँटें",
    "wash clothes and bedding in hot water": "कपड़े और बिस्तर गर्म पानी में धोएँ",
    "wash clothes, towels and bedding in hot water": "कपड़े, तौलिये और बिस्तर गर्म पानी में धोएँ",
    "seal items that cannot be washed in a bag for several days": "जो चीज़ें धुल नहीं सकतीं उन्हें कुछ दिनों के लिए थैले में बंद रखें",
    "avoid close skin contact until checked": "जाँच होने तक नज़दीकी त्वचा संपर्क से बचें",
    "wash twice a day with a mild cleanser": "दिन में दो बार हल्के क्लींज़र से धोएँ",
    "do not squeeze or pick spots": "दानों को न दबाएँ और न नोचें",
    "choose oil-free, non-comedogenic products": "तेल-रहित, रोमछिद्र बंद न करने वाले प्रोडक्ट चुनें",
    "change pillowcases often": "तकिये का कवर अक्सर बदलें",
    "moisturise thick areas daily": "मोटे हिस्सों पर रोज़ नमी दें",
    "short sun exposure without burning": "थोड़ी देर धूप लें, पर जलने न दें",
    "avoid scratching or injuring the skin": "खुजलाने या त्वचा को चोट पहुँचाने से बचें",
    "manage stress": "तनाव का ध्यान रखें",
    "protect white patches from sunburn with clothing and shade": "सफ़ेद दागों को कपड़ों और छाँव से धूप में जलने से बचाएँ",
    "be gentle with the skin; injury can trigger new patches": "त्वचा के साथ नरमी बरतें; चोट से नए दाग बन सकते हैं",
    "support groups can help": "सहायता समूह मदद कर सकते हैं",
    "photograph it to compare over time": "समय के साथ तुलना के लिए इसकी फ़ोटो लें",
    "protect skin from strong sun": "त्वचा को तेज़ धूप से बचाएँ",
    "check your skin once a month": "महीने में एक बार अपनी त्वचा जाँचें",
    "do not try to remove or burn it at home": "इसे घर पर हटाने या जलाने की कोशिश न करें",
    "photograph it with a coin for size": "आकार के लिए सिक्के के साथ इसकी फ़ोटो लें",
    "retake the photo in daylight, 15-20 cm away": "दिन की रोशनी में 15-20 सेमी दूर से फिर फ़ोटो लें",
    "keep the area clean and avoid new products until checked": "जाँच होने तक उस हिस्से को साफ़ रखें और नए प्रोडक्ट से बचें",
}
HELP = [
    ("A clearer photo in daylight, 15–20 cm away", "दिन की रोशनी में 15–20 सेमी दूर से ली गई साफ़ फ़ोटो"),
    ("A photo of another affected area", "किसी दूसरे प्रभावित हिस्से की फ़ोटो"),
    ("A doctor's examination in person", "डॉक्टर द्वारा आमने-सामने जाँच"),
    ("Noting whether it changes over the next two weeks", "अगले दो हफ़्तों में यह बदलता है या नहीं, यह लिखकर रखना"),
    ("Checking whether others at home itch at night", "यह देखना कि घर में दूसरों को रात में खुजली होती है या नहीं"),
    ("Remembering any new product used before it started", "शुरू होने से पहले इस्तेमाल किया गया कोई नया प्रोडक्ट याद करना"),
]
DUR_EN = {"lt_1w": "less than a week", "1_4w": "one to four weeks", "1_6m": "one to six months", "gt_6m": "more than six months"}
DUR_HI = {"lt_1w": "एक हफ़्ते से कम", "1_4w": "एक से चार हफ़्ते", "1_6m": "एक से छह महीने", "gt_6m": "छह महीने से ज़्यादा"}
ITCH_EN = ["no itching", "mild itching", "moderate itching", "severe itching"]
ITCH_HI = ["कोई खुजली नहीं", "हल्की खुजली", "मध्यम खुजली", "बहुत ज़्यादा खुजली"]
SITE_EN = {"face": "face", "scalp": "scalp", "neck": "neck", "chest": "chest", "back": "back", "abdomen": "stomach", "arm": "arm",
           "hand": "hand", "leg": "leg", "foot": "foot", "groin": "groin", "nails": "nails", "other": "skin"}
SITE_HI = {"face": "चेहरे", "scalp": "सिर", "neck": "गर्दन", "chest": "छाती", "back": "पीठ", "abdomen": "पेट", "arm": "बाँह",
           "hand": "हाथ", "leg": "पैर", "foot": "पैर के पंजे", "groin": "जांघ के जोड़", "nails": "नाखून", "other": "त्वचा"}
DISPLAY_HI = {"eczema_atopic": "एक्ज़िमा", "contact_dermatitis": "कॉन्टैक्ट डर्मेटाइटिस", "seborrheic_dermatitis": "सेबोरिक डर्मेटाइटिस",
              "tinea": "फंगल संक्रमण (दाद)", "scabies": "स्केबीज़", "acne": "मुंहासे", "psoriasis": "सोरायसिस", "vitiligo": "विटिलिगो",
              "benign_lesion": "हानिरहित उभार", "suspicious_lesion": "डॉक्टर को दिखाने लायक दाग", "other": "अन्य"}
