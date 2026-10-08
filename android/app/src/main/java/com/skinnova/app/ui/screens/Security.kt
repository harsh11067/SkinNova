package com.skinnova.app.ui.screens

import androidx.biometric.BiometricManager
import androidx.biometric.BiometricManager.Authenticators.BIOMETRIC_STRONG
import androidx.biometric.BiometricPrompt
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableLongStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import androidx.fragment.app.FragmentActivity
import com.skinnova.app.R
import com.skinnova.app.data.Profile
import com.skinnova.app.model.Enums
import com.skinnova.app.security.AppLock
import com.skinnova.app.security.Unlock
import com.skinnova.app.ui.SessionViewModel
import com.skinnova.app.ui.components.Chip
import com.skinnova.app.ui.components.Overline
import com.skinnova.app.ui.components.PixelImage
import com.skinnova.app.ui.components.PrimaryButton
import com.skinnova.app.ui.components.SnCard
import com.skinnova.app.ui.components.Toggle
import com.skinnova.app.ui.components.TopBar
import com.skinnova.app.ui.theme.LocalSn
import com.skinnova.app.ui.theme.SnType
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

fun biometricAvailable(ctx: android.content.Context) =
    BiometricManager.from(ctx).canAuthenticate(BIOMETRIC_STRONG) == BiometricManager.BIOMETRIC_SUCCESS

/** Fingerprint (Class 3 / BIOMETRIC_STRONG only: no 2D face unlock) → [onOk]. */
fun promptBiometric(act: FragmentActivity, title: String, cancel: String, onOk: () -> Unit) {
    if (!biometricAvailable(act)) return
    BiometricPrompt(act, ContextCompat.getMainExecutor(act), object : BiometricPrompt.AuthenticationCallback() {
        override fun onAuthenticationSucceeded(result: BiometricPrompt.AuthenticationResult) = onOk()
    }).authenticate(BiometricPrompt.PromptInfo.Builder().setTitle(title).setNegativeButtonText(cancel).setAllowedAuthenticators(BIOMETRIC_STRONG).build())
}

/** PIN dots + 3×4 number pad (48 dp+ keys). */
@Composable
private fun PinPad(pin: String, length: Int, onDigit: (Char) -> Unit, onBack: () -> Unit, extra: (@Composable () -> Unit)? = null) {
    val sn = LocalSn.current
    Column(horizontalAlignment = Alignment.CenterHorizontally) {
        Row(horizontalArrangement = Arrangement.spacedBy(14.dp), modifier = Modifier.semantics { contentDescription = "${pin.length} digits" }) {
            repeat(maxOf(length, pin.length)) { i ->
                Box(Modifier.size(14.dp).clip(CircleShape).background(if (i < pin.length) sn.acc else Color.Transparent).border(1.5.dp, sn.acc, CircleShape))
            }
        }
        Spacer(Modifier.height(28.dp))
        listOf("123", "456", "789", "x0<").forEach { row ->
            Row(horizontalArrangement = Arrangement.spacedBy(18.dp)) {
                row.forEach { k ->
                    val mod = Modifier.size(72.dp).clip(CircleShape)
                    when (k) {
                        'x' -> Box(mod, contentAlignment = Alignment.Center) { extra?.invoke() }
                        '<' -> Box(mod.clickable(role = Role.Button, onClick = onBack).semantics { contentDescription = "Delete" }, contentAlignment = Alignment.Center) {
                            Text("⌫", color = sn.ink, fontSize = 22.sp) }
                        else -> Box(mod.background(sn.surf).border(1.dp, sn.line2, CircleShape).clickable(role = Role.Button) { onDigit(k) },
                            contentAlignment = Alignment.Center) { Text("$k", style = SnType.headline, color = sn.ink) }
                    }
                }
            }
            Spacer(Modifier.height(14.dp))
        }
    }
}

/**
 * Covers every route while the app is locked (MainActivity): opaque, takes all touches, nothing underneath is readable
 * by accessibility services. Unlock by PIN, or fingerprint if turned on.
 */
@Composable
fun LockScreen(lock: AppLock, onForgot: () -> Unit) {
    val sn = LocalSn.current
    val ctx = LocalContext.current
    val bio by lock.biometric.collectAsState()
    var pin by remember { mutableStateOf("") }
    var msg by remember { mutableStateOf<String?>(null) }
    var until by remember { mutableLongStateOf(lock.lockedOutUntil()) }
    var nowMs by remember { mutableLongStateOf(System.currentTimeMillis()) }
    var forgot by remember { mutableStateOf(false) }
    val bioTitle = stringResource(R.string.lock_bio_title); val usePin = stringResource(R.string.lock_use_pin)
    fun askBio() { (ctx as? FragmentActivity)?.let { promptBiometric(it, bioTitle, usePin) { lock.unlockWithBiometric() } } }
    LaunchedEffect(Unit) { if (bio && biometricAvailable(ctx)) askBio() }
    LaunchedEffect(until) { while (until > System.currentTimeMillis()) { nowMs = System.currentTimeMillis(); delay(500) }; until = 0; msg = null }
    val wrong = stringResource(R.string.lock_wrong); val wait = stringResource(R.string.lock_wait)
    Box(Modifier.fillMaxSize().background(sn.bgBrush).pointerInput(Unit) { detectTapGestures { } }) {
        Column(Modifier.fillMaxSize().statusBarsPadding().navigationBarsPadding().padding(24.dp), horizontalAlignment = Alignment.CenterHorizontally) {
            Spacer(Modifier.height(40.dp))
            PixelImage(R.drawable.px_gem, Modifier.size(56.dp), ContentScale.Fit)
            Spacer(Modifier.height(16.dp))
            Text(stringResource(R.string.lock_title), style = SnType.headline, color = sn.ink)
            Spacer(Modifier.height(8.dp))
            Text(when { until > nowMs -> wait.format(((until - nowMs) / 1000) + 1); msg != null -> msg!!; else -> stringResource(R.string.lock_sub) },
                style = SnType.body, color = if (msg != null || until > nowMs) sn.urgent else sn.mut)
            Spacer(Modifier.weight(1f))
            PinPad(pin, 4, onDigit = { d ->
                if (until > System.currentTimeMillis() || pin.length >= 8) return@PinPad
                pin += d
                if (pin.length >= 4) when (val r = lock.check(pin)) {
                    Unlock.Ok -> pin = ""
                    is Unlock.Wrong -> if (pin.length == 8) { msg = wrong.format(r.triesLeft); pin = "" }
                    is Unlock.LockedOut -> { until = r.untilMs; pin = "" }
                }
            }, onBack = { pin = pin.dropLast(1) }, extra = if (bio && biometricAvailable(ctx)) ({
                Box(Modifier.size(72.dp).clip(CircleShape).clickable(role = Role.Button) { askBio() }.semantics { contentDescription = bioTitle },
                    contentAlignment = Alignment.Center) { Text("☝", fontSize = 24.sp, color = sn.accT) }
            }) else null)
            Spacer(Modifier.height(6.dp))
            TextButton({ forgot = true }) { Text(stringResource(R.string.lock_forgot), color = sn.mut, style = SnType.caption) }
            Spacer(Modifier.height(12.dp))
        }
    }
    if (forgot) AlertDialog(onDismissRequest = { forgot = false },
        title = { Text(stringResource(R.string.lock_forgot)) }, text = { Text(stringResource(R.string.lock_reset_body)) },
        confirmButton = { TextButton({ forgot = false; onForgot() }) { Text(stringResource(R.string.lock_reset_ok), color = Color(0xFFE0787A)) } },
        dismissButton = { TextButton({ forgot = false }) { Text(stringResource(R.string.cancel)) } })
}

/** Choose a PIN (twice). Matching PINs → [onDone]. 4–8 digits; 1111 / 1234-style PINs are refused. */
@Composable
fun SetPinScreen(onDone: (String) -> Unit, onBack: () -> Unit) {
    val sn = LocalSn.current
    var first by remember { mutableStateOf<String?>(null) }
    var pin by remember { mutableStateOf("") }
    var msg by remember { mutableStateOf<Int?>(null) }
    Column(Modifier.fillMaxSize().statusBarsPadding().navigationBarsPadding().padding(horizontal = 18.dp, vertical = 8.dp), horizontalAlignment = Alignment.CenterHorizontally) {
        TopBar(stringResource(R.string.pin_title), onBack)
        Spacer(Modifier.height(24.dp))
        Text(stringResource(if (first == null) R.string.pin_choose else R.string.pin_confirm), style = SnType.titleL, color = sn.ink)
        Spacer(Modifier.height(8.dp))
        Text(stringResource(msg ?: R.string.pin_rules), style = SnType.body, color = if (msg != null) sn.urgent else sn.mut)
        Spacer(Modifier.weight(1f))
        PinPad(pin, 4, onDigit = { if (pin.length < 8) { pin += it; msg = null } }, onBack = { pin = pin.dropLast(1) })
        Spacer(Modifier.height(8.dp))
        PrimaryButton(stringResource(if (first == null) R.string.pin_next else R.string.pin_save), Modifier.fillMaxWidth(), enabled = pin.length >= 4) {
            val f = first
            if (f == null) {
                when (AppLock.validPin(pin)) { null -> { first = pin; pin = "" }; "simple" -> { msg = R.string.pin_simple; pin = "" }; else -> { msg = R.string.pin_rules; pin = "" } }
            } else if (f == pin) onDone(pin) else { msg = R.string.pin_mismatch; first = null; pin = "" }
        }
        Spacer(Modifier.height(16.dp))
    }
}

/** Profile form (all optional; stored encrypted on this phone). */
@OptIn(ExperimentalLayoutApi::class)
@Composable
fun ProfileEditScreen(vm: SessionViewModel, onDone: () -> Unit) {
    val sn = LocalSn.current
    val c = vm.c
    val cur by c.profiles.profile.collectAsState()
    var p by remember { mutableStateOf(cur) }
    val scope = rememberCoroutineScope()
    Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).statusBarsPadding().navigationBarsPadding().imePadding()
        .padding(horizontal = 18.dp, vertical = 8.dp)) {
        TopBar(stringResource(R.string.pe_title), onDone)
        Text(stringResource(R.string.pe_sub), style = SnType.body, color = sn.mut)
        Overline(stringResource(R.string.pe_name))
        Field(p.name, stringResource(R.string.pe_name_hint)) { p = p.copy(name = it.take(40)) }
        Overline(stringResource(R.string.q_age))
        FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Enums.AGE_BANDS.forEach { a -> Chip(stringResource(ageRes(a)), p.ageBand == a, { p = p.copy(ageBand = a) }) }
        }
        Overline(stringResource(R.string.pe_sex))
        FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            (Profile.SEXES + listOf(null)).forEach { s ->
                Chip(stringResource(when (s) { "female" -> R.string.pe_female; "male" -> R.string.pe_male; "other" -> R.string.pe_other; else -> R.string.pe_prefer_not }),
                    p.sex == s, { p = p.copy(sex = s, pregnant = if (s == "male") false else p.pregnant) })
            }
        }
        Overline(stringResource(R.string.q_tone))
        FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Enums.SKIN_TONES.forEach { t -> Chip(stringResource(toneRes(t)), p.skinTone == t, { p = p.copy(skinTone = t) }) }
        }
        if (p.sex != "male") {
            Spacer(Modifier.height(12.dp))
            Row(Modifier.fillMaxWidth().heightIn(min = 56.dp).clickable(role = Role.Switch) { p = p.copy(pregnant = !p.pregnant) }, verticalAlignment = Alignment.CenterVertically) {
                Text(stringResource(R.string.pe_pregnant), style = SnType.bodyL, color = sn.ink, modifier = Modifier.weight(1f)); Toggle(p.pregnant)
            }
        }
        Overline(stringResource(R.string.pe_conditions))
        Text(stringResource(R.string.pe_conditions_sub), style = SnType.micro, color = sn.mut)
        Spacer(Modifier.height(8.dp))
        FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Profile.CONDITIONS.forEach { k ->
                val on = k in p.conditions
                Chip(stringResource(conditionRes(k)), on, { p = p.copy(conditions = if (on) p.conditions - k else p.conditions + k) })
            }
        }
        Overline(stringResource(R.string.pe_allergies))
        Field(p.allergies, stringResource(R.string.pe_allergies_hint)) { p = p.copy(allergies = it.take(120)) }
        Spacer(Modifier.height(20.dp))
        PrimaryButton(stringResource(R.string.pe_save), Modifier.fillMaxWidth()) { scope.launch { c.profiles.save(p); onDone() } }
        Spacer(Modifier.height(10.dp))
        Box(Modifier.fillMaxWidth().clip(RoundedCornerShape(16.dp)).background(sn.surf2).padding(12.dp)) {
            Text(stringResource(R.string.pe_privacy), style = SnType.micro, color = sn.mut)
        }
        Spacer(Modifier.height(24.dp))
    }
}

@Composable
private fun Field(value: String, hint: String, onChange: (String) -> Unit) {
    val sn = LocalSn.current
    BasicTextField(value, onChange, Modifier.fillMaxWidth().clip(RoundedCornerShape(16.dp)).background(sn.surf).border(1.dp, sn.line, RoundedCornerShape(16.dp)).padding(14.dp),
        textStyle = SnType.bodyL.copy(color = sn.ink), cursorBrush = SolidColor(sn.acc), singleLine = true,
        decorationBox = { inner -> if (value.isEmpty()) Text(hint, style = SnType.bodyL, color = sn.mut); inner() })
}

fun ageRes(a: String) = when (a) { "lt_12" -> R.string.age_lt_12; "12_17" -> R.string.age_12_17; "18_39" -> R.string.age_18_39; "40_59" -> R.string.age_40_59; else -> R.string.age_60_plus }
fun toneRes(t: String) = when (t) { "fitz_1_2" -> R.string.tone_fitz_1_2; "fitz_3_4" -> R.string.tone_fitz_3_4; "fitz_5_6" -> R.string.tone_fitz_5_6; else -> R.string.tone_unknown }
fun conditionRes(k: String) = when (k) { "diabetes" -> R.string.pc_diabetes; "weak_immunity" -> R.string.pc_weak_immunity; "asthma" -> R.string.pc_asthma
    "stomach_ulcer" -> R.string.pc_stomach_ulcer; else -> R.string.pc_kidney_liver }

/** Profile → Security section: app lock, fingerprint, auto-lock delay. */
@Composable
fun SecuritySection(lock: AppLock, onSetPin: () -> Unit) {
    val sn = LocalSn.current
    val ctx = LocalContext.current
    val bio by lock.biometric.collectAsState()
    val grace by lock.graceMs.collectAsState()
    var enabled by remember { mutableStateOf(lock.enabled) }
    var confirmOff by remember { mutableStateOf(false) }
    SnCard(framed = false) {
        Column {
            Row(Modifier.fillMaxWidth().heightIn(min = 62.dp).clickable(role = Role.Switch) { if (enabled) confirmOff = true else onSetPin() },
                verticalAlignment = Alignment.CenterVertically) {
                Column(Modifier.weight(1f)) {
                    Text(stringResource(R.string.sec_lock), style = SnType.bodyL, color = sn.ink)
                    Text(stringResource(R.string.sec_lock_sub), style = SnType.micro, color = sn.mut)
                }
                Toggle(enabled)
            }
            if (enabled) {
                if (biometricAvailable(ctx)) Row(Modifier.fillMaxWidth().heightIn(min = 56.dp).clickable(role = Role.Switch) { lock.setBiometric(!bio) },
                    verticalAlignment = Alignment.CenterVertically) {
                    Text(stringResource(R.string.sec_bio), style = SnType.bodyL, color = sn.ink, modifier = Modifier.weight(1f)); Toggle(bio)
                }
                Text(stringResource(R.string.sec_after), style = SnType.bodyL, color = sn.ink, modifier = Modifier.padding(top = 10.dp))
                Row(horizontalArrangement = Arrangement.spacedBy(6.dp), modifier = Modifier.padding(vertical = 8.dp)) {
                    listOf(1_000L to R.string.sec_now, 60_000L to R.string.sec_1m, 300_000L to R.string.sec_5m).forEach { (ms, l) ->
                        Chip(stringResource(l), grace == ms, { lock.setGrace(ms) }, Modifier.weight(1f))
                    }
                }
                Row(Modifier.fillMaxWidth().heightIn(min = 56.dp).clickable(role = Role.Button, onClick = onSetPin), verticalAlignment = Alignment.CenterVertically) {
                    Text(stringResource(R.string.sec_change), style = SnType.bodyL, color = sn.ink, modifier = Modifier.weight(1f)); Text("›", color = sn.mut)
                }
            }
        }
    }
    if (confirmOff) AlertDialog(onDismissRequest = { confirmOff = false }, text = { Text(stringResource(R.string.sec_off_confirm)) },
        confirmButton = { TextButton({ confirmOff = false; lock.disable(); enabled = false }) { Text(stringResource(R.string.sec_off)) } },
        dismissButton = { TextButton({ confirmOff = false }) { Text(stringResource(R.string.cancel)) } })
}
