package com.skinnova.app.di

import android.content.Context
import com.skinnova.app.data.ImageStore
import com.skinnova.app.data.Settings
import com.skinnova.app.data.SnDb
import com.skinnova.app.ml.AnalysisPipeline
import com.skinnova.app.ml.CvClassifier
import com.skinnova.app.ml.EngineHolder
import com.skinnova.app.ml.GemmaEngine
import com.skinnova.app.ml.IntakeValidator
import com.skinnova.app.ml.Llm
import com.skinnova.app.ml.OutputParser
import com.skinnova.app.ml.PromptBuilder
import com.skinnova.app.model.Labels
import com.skinnova.app.model.SnJson
import com.skinnova.app.safety.ContentGuards
import com.skinnova.app.setup.ModelManager

/** Manual DI (architecture §2): process-wide singletons. */
class AppContainer(val ctx: Context) {
    private fun asset(path: String) = ctx.assets.open(path).bufferedReader().use { it.readText() }

    val settings = Settings(ctx)
    val labels: Labels = SnJson.decodeFromString(Labels.serializer(), asset("labels.json"))
    val guards = ContentGuards(ContentGuards.parseTerms(asset("safety/rx_terms.txt")))
    val intakeTopics = IntakeValidator.parseTopics(asset("safety/intake_topics.json"))
    val prompts = PromptBuilder(::asset)
    val parser = OutputParser(labels.keys, guards)
    val models = ModelManager(ctx)
    val engineHolder = EngineHolder(ctx, models)
    var llm: Llm = GemmaEngine(engineHolder, models)
    val cv = CvClassifier(ctx, labels.keys)
    val db by lazy { SnDb.build(ctx) }
    val images by lazy { ImageStore(ctx) }

    fun pipeline() = AnalysisPipeline(labels, prompts, parser, guards, llm, { engineHolder.isLoaded },
        { models.activeModel()?.sha256?.take(12) ?: "" })
}
