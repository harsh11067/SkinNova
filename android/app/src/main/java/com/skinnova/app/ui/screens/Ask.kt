package com.skinnova.app.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.skinnova.app.R
import kotlinx.coroutines.launch
import com.skinnova.app.ui.SessionViewModel
import com.skinnova.app.ui.components.SnCard
import com.skinnova.app.ui.components.SpeakButton
import com.skinnova.app.ui.theme.LocalSn
import com.skinnova.app.ui.theme.SnType

/**
 * Ask SkinNova: follow-up questions about this result, answered on the phone by the fine-tuned Gemma
 * (prompts/chat_*.txt). Answers appear only after the safety checks (safety/ChatSafety.kt); a danger sign in the
 * question always shows the fixed "get care today" line, whatever the model writes.
 */
@OptIn(ExperimentalLayoutApi::class)
@Composable
fun AskCard(vm: SessionViewModel, tts: Boolean, lang: String) {
    if (!vm.c.llm.available) return
    val sn = LocalSn.current
    val turns by vm.chat.collectAsState()
    var q by remember { mutableStateOf("") }
    val busy = turns.lastOrNull()?.answer == null && turns.isNotEmpty()
    fun send(text: String) { if (!busy && text.isNotBlank()) { vm.ask(text); q = "" } }
    val ctx = androidx.compose.ui.platform.LocalContext.current
    val scope = androidx.compose.runtime.rememberCoroutineScope()
    var listening by remember { mutableStateOf(false) }
    fun listen() {
        if (listening) return
        listening = true
        scope.launch { val heard = vm.listenQuestion { }; listening = false; if (heard != null) q = heard.take(300) }
    }
    val micPerm = androidx.activity.compose.rememberLauncherForActivityResult(
        androidx.activity.result.contract.ActivityResultContracts.RequestPermission()) { if (it) listen() }
    SnCard(framed = true) {
        Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(stringResource(R.string.ask_title), style = SnType.title, color = sn.ink, modifier = Modifier.weight(1f))
                com.skinnova.app.ui.components.SmallTag(stringResource(R.string.ask_badge), sn.accT)
            }
            if (listening) Text(stringResource(R.string.ask_listening), style = SnType.caption, color = sn.accT)
            Text(stringResource(R.string.ask_sub), style = SnType.caption, color = sn.mut)
            turns.forEach { t ->
                Box(Modifier.fillMaxWidth().padding(start = 40.dp), contentAlignment = Alignment.CenterEnd) {
                    Text(t.question, style = SnType.body, color = sn.ink, modifier = Modifier.clip(RoundedCornerShape(14.dp)).background(sn.surf2).padding(10.dp))
                }
                if (t.danger) Text("⚠  " + stringResource(R.string.ask_danger), style = SnType.body, color = sn.urgent)
                when {
                    t.answer == null -> Text(stringResource(R.string.ask_writing), style = SnType.caption, color = sn.mut)
                    t.withheld -> Text(stringResource(R.string.ask_withheld), style = SnType.body, color = sn.mut)
                    else -> Row(verticalAlignment = Alignment.Top) {
                        Text(t.answer, style = SnType.body, color = sn.ink, modifier = Modifier.weight(1f))
                        if (tts) SpeakButton({ t.answer }, lang)
                    }
                }
            }
            if (turns.isEmpty()) FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                listOf(R.string.ask_s_contagious, R.string.ask_s_avoid, R.string.ask_s_heal, R.string.ask_s_doctor).forEach { id ->
                    val s = stringResource(id)
                    Box(Modifier.heightIn(min = 48.dp).clip(RoundedCornerShape(20.dp)).border(1.dp, sn.line2, RoundedCornerShape(20.dp))
                        .clickable(role = Role.Button) { send(s) }.padding(horizontal = 14.dp), contentAlignment = Alignment.Center) {
                        Text(s, style = SnType.caption, color = sn.accT)
                    }
                }
            }
            Row(verticalAlignment = Alignment.CenterVertically) {
                BasicTextField(q, { q = it.take(300) }, Modifier.weight(1f).clip(RoundedCornerShape(16.dp)).background(sn.surf)
                    .border(1.dp, sn.line, RoundedCornerShape(16.dp)).padding(horizontal = 14.dp),
                    textStyle = com.skinnova.app.ui.components.fieldTextStyle.copy(color = sn.ink), cursorBrush = SolidColor(sn.acc), singleLine = true,
                    keyboardOptions = KeyboardOptions(imeAction = ImeAction.Send), keyboardActions = KeyboardActions(onSend = { send(q) }),
                    decorationBox = { inner -> if (q.isEmpty()) Text(stringResource(R.string.ask_hint), style = com.skinnova.app.ui.components.fieldTextStyle, color = sn.mut); inner() })
                Spacer(Modifier.width(8.dp))
                val micLabel = stringResource(R.string.ask_speak)
                Box(Modifier.size(48.dp).clip(RoundedCornerShape(16.dp)).background(if (listening) sn.acc else sn.surf2)
                    .semantics { contentDescription = micLabel }.clickable(enabled = !busy, role = Role.Button) {
                        if (androidx.core.content.ContextCompat.checkSelfPermission(ctx, android.Manifest.permission.RECORD_AUDIO) ==
                            android.content.pm.PackageManager.PERMISSION_GRANTED) listen() else micPerm.launch(android.Manifest.permission.RECORD_AUDIO)
                    }, contentAlignment = Alignment.Center) { Text("🎙", fontSize = 16.sp) }
                Spacer(Modifier.width(8.dp))
                val sendLabel = stringResource(R.string.ask_send)
                Box(Modifier.size(48.dp).clip(RoundedCornerShape(16.dp)).background(if (busy || q.isBlank()) sn.surf2 else sn.acc)
                    .semantics { contentDescription = sendLabel }.clickable(enabled = !busy && q.isNotBlank(), role = Role.Button) { send(q) },
                    contentAlignment = Alignment.Center) { Text("➤", color = sn.onAcc, fontSize = 16.sp) }
            }
            Text(stringResource(R.string.ask_note), style = SnType.micro, color = sn.mut)
        }
    }
}
