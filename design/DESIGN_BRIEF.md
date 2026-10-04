# SkinNova — Design brief (paste into Claude Design)

Design a calm, trustworthy Android app (phone, 412×915, light + dark) called **SkinNova**. It gives *preliminary educational* information about skin from a photo, works offline, supports voice input in Hindi/English, and tracks a skin spot over time. Tone: clinical-calm and warm, never alarming or playful. Imagery and illustrations must represent brown and dark skin tones first (primary users: India).

## Screens (deliver each in light + dark)
1. **Onboarding (3 cards) + Disclaimer**: what it does · private & offline (no internet needed) · not a diagnosis → "I understand". Language picker (English / हिन्दी / beta languages).
2. **Model setup**: one-time import of a 2.6 GB file: "Import model file" (primary), storage-needed note, progress with % → "Verifying…" → "Ready ✓". Error states: not enough space, wrong file.
3. **Home**: big "Check my skin" button, "My tracked spots" list (thumbnail, name, next re-check date), history entry, settings.
4. **Capture**: camera viewfinder with circular framing guide, live hints ("Move closer", "More light", "Hold steady"), gallery button, quality warning sheet ("Photo looks blurry — retake / use anyway").
5. **Questionnaire**: one question per card with progress dots; chips for options, 0–3 slider for itch/pain, front/back body silhouette picker. A prominent **mic button "Speak instead / बोलिए"**.
6. **Voice intake**: recording state (waveform, 30 s timer, stop), "Here's what we heard" editable transcript, then a **pre-filled form where each filled answer shows a small quote chip of what the user said** ("‘teen hafte se’ → 1–4 weeks"), unfilled answers highlighted "Please answer". Confirm button.
7. **Analyzing**: photo thumbnail, step list (Checking photo → Image model → Writing explanation → Translating), streaming preview text, Cancel.
8. **Result**, top to bottom: red-flag banner (only if present) → triage card (LOW green / MODERATE amber / HIGH orange / URGENT red; icon + words, never colour alone) → "Possible categories" (3 rows, likelihood pill: higher / possible / less likely, one-line why) → uncertainty meter + reasons → explanation (🔊 read aloud, EN/हिं toggle) → What would help → General care info → disclaimer footer → actions: **Track this spot**, Save, Retake. "Basic mode" variant (AI explanation unavailable).
9. **Track this spot** sheet: name the spot, body site, reminder (3/7/14/30 days), "Add a coin next to the spot for accurate size" tip with illustration.
10. **Re-capture with ghost overlay**: live camera with the previous photo at ~35% opacity aligned on top, alignment indicator (red→amber→green ring), coin-detected badge.
11. **Timeline**: horizontal photo strip with dates, small sparkline charts (size change %, colour contrast change), confidence label (OK / Low), Gemma's 2–4-sentence change summary, tier history chips, "Export Doctor Visit Summary (PDF)".
12. **Doctor Visit Summary preview** (A4 page look): header, spot info, table of captures with photos and metrics, latest answers, questions to ask the doctor, disclaimer.
13. **History**: list (thumbnail, date, top category, tier), delete-all (confirmation).
14. **Settings / About**: language, voice read-aloud, history on/off, coin size, delete everything, model info (version, size), data sources & licenses, beta labels.

## Constraints
- Font: Inter or Noto Sans (must render Devanagari well; use Noto Sans Devanagari for Hindi). Touch targets ≥ 48 dp, WCAG AA contrast, works at 1.5× font scale.
- No web-only effects (backdrop blur, CSS filters on photos, heavy gradients over photos).
- Never a single bold diagnosis headline. Category names are larger than any number.
- Deliver: one PNG per screen (light + dark), the HTML/CSS, and `tokens.json` with colors (light/dark, incl. 4 triage colours + on-colours), typography scale, spacing scale, radii, elevation.
