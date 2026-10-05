package com.skinnova.app.setup

import android.content.Context

/** offline flavor: no network code at all (and no INTERNET permission). Import via file picker / adb only. */
object ModelDownload {
    const val available = false
    @Suppress("UNUSED_PARAMETER")
    suspend fun download(ctx: Context, models: ModelManager, onState: (ImportState) -> Unit): ImportState =
        ImportState.Error(ImportState.Error.Kind.IO, "download not available in the offline build")
}
