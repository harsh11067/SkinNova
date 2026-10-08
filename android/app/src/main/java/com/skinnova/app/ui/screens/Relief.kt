package com.skinnova.app.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.Alignment
import androidx.compose.ui.draw.clip
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import com.skinnova.app.R
import com.skinnova.app.care.Blocked
import com.skinnova.app.care.Caution
import com.skinnova.app.care.ReliefPlanner
import com.skinnova.app.model.FinalResult
import com.skinnova.app.model.Tier
import com.skinnova.app.ui.SessionViewModel
import com.skinnova.app.ui.components.SnCard
import com.skinnova.app.ui.components.SpeakButton
import com.skinnova.app.ui.theme.LocalSn
import com.skinnova.app.ui.theme.SnType

/** Result §6b: home care, food & lifestyle, pharmacy relief — from assets/care/relief.json via [ReliefPlanner]. */
@Composable
fun ReliefCard(vm: SessionViewModel, r: FinalResult, tier: Tier, tts: Boolean, lang: String) {
    val sn = LocalSn.current
    val profile by vm.c.profiles.profile.collectAsState()
    val top = r.output.possibleCategories.firstOrNull() ?: return
    val plan = remember(r.createdAt, profile, tier) {
        runCatching {
            ReliefPlanner.plan(vm.c.relief, top.key, top.likelihood, r.cvTop3.firstOrNull { it.key == top.key }?.p ?: 0.0,
                r.output.possibleCategories.map { it.key }.toSet(), tier, r.answers, profile)
        }.getOrNull()
    } ?: return
    val title = stringResource(R.string.rl_title)
    val hHome = stringResource(R.string.rl_home); val hFood = stringResource(R.string.rl_food); val hPharm = stringResource(R.string.rl_pharmacy)
    val blocked = plan.blocked?.let { stringResource(if (it == Blocked.DOCTOR_FIRST) R.string.rl_doctor_first else R.string.rl_child) }
    val cautions = plan.cautions.map { c ->
        when (c) {
            Caution.PHARMACIST -> stringResource(R.string.rl_c_pharmacist)
            Caution.CONFIRM_FIRST -> stringResource(R.string.rl_c_confirm)
            Caution.PREGNANT -> stringResource(R.string.rl_c_pregnant)
            Caution.ALLERGIES -> stringResource(R.string.rl_c_allergies, profile.allergies)
            Caution.INFECTION_RISK -> stringResource(R.string.rl_c_infection)
        }
    }
    val spoken = buildString {
        append(title).append(". ")
        plan.contagious?.let { append(it.get(lang)).append(" ") }
        if (plan.home.isNotEmpty()) append(hHome).append(". ").append(plan.home.joinToString(" ") { it.get(lang) }).append(" ")
        if (plan.food.isNotEmpty()) append(hFood).append(". ").append(plan.food.joinToString(" ") { it.get(lang) }).append(" ")
        append(hPharm).append(". ")
        if (blocked != null) append(blocked) else append(plan.pharmacy.joinToString(" ") { it.get(lang) }).append(" ").append(cautions.joinToString(" "))
    }
    SnCard(framed = false) {
        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(title, style = SnType.title, color = sn.ink, modifier = Modifier.weight(1f))
                if (tts) SpeakButton({ spoken }, lang)
            }
            plan.contagious?.let { Section(stringResource(R.string.rl_contagious), listOf(it.get(lang))) }
            Section(hHome, plan.home.map { it.get(lang) })
            Section(hFood, plan.food.map { it.get(lang) })
            Text(hPharm, style = SnType.label, color = sn.accT, modifier = Modifier.padding(top = 4.dp))
            if (blocked != null) Box(Modifier.fillMaxWidth().clip(RoundedCornerShape(14.dp)).background(sn.surf2).padding(12.dp)) {
                Text(blocked, style = SnType.body, color = sn.ink)
            } else {
                plan.pharmacy.forEach { Bullet(it.get(lang)) }
                Spacer(Modifier.height(2.dp))
                cautions.forEach { Text("ⓘ  $it", style = SnType.caption, color = sn.mut) }
            }
            Text(stringResource(R.string.ins_sources) + ": " + plan.sources.joinToString(" · "), style = SnType.micro, color = sn.mut,
                modifier = Modifier.padding(top = 4.dp))
        }
    }
}

@Composable
private fun Section(title: String, items: List<String>) {
    if (items.isEmpty()) return
    val sn = LocalSn.current
    Text(title, style = SnType.label, color = sn.accT, modifier = Modifier.padding(top = 4.dp))
    items.forEach { Bullet(it) }
}

@Composable
private fun Bullet(t: String) {
    val sn = LocalSn.current
    Row { Text("•  ", color = sn.accT, style = SnType.body); Text(t, style = SnType.body, color = sn.mut) }
}
