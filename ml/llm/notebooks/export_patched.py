"""`litert-torch export_hf` (0.9.4) with three memory fixes for Gemma 4 E2B on Kaggle's 32 GB CPU machine.

Export v4–v6 were OOM-killed (rc -9). v6 (with --experimental_lightweight_conversion) finished the text model and the
embedder, then died in "Export per_layer_embedder model > Create MLIR Module" (min. MemAvailable seen 5.0 GB of 32 GB):
1. upstream `export_additional_models_impl` calls `converter.convert(strict_export=False)` WITHOUT the lightweight flag,
   although that table (262,144 x 8,960 fp32 = 9.4 GB) is traced for two signatures → pass the flag through;
2. after the text prefill/decode model is written, the decoder-layer and audio-tower weights are not used by the
   remaining steps (embedder, per-layer embedder, vision encoder, tokenizer) → release them (~2 B params, ~8 GB fp32).
v7 then got through the per-layer embedder and died in "Export vision encoder models > vision_adapter > Create MLIR
Module" (min. MemAvailable 1.4 GB): the 9.4 GB per-layer table and the 1.6 GB token embedding were still held →
3. after the per-layer embedder is written, release the text embeddings too (the vision step uses only vision_tower +
   embed_vision), and hand freed heap back to the OS (glibc malloc_trim) after each release.
Nothing about the exported graphs changes. Usage: python export_patched.py export_hf <src> <out> --flags...
"""
import ctypes
import dataclasses
import gc
import os

import torch
from litert_torch import cli
from litert_torch.generative.export_hf.core import export_lib as E


def _additional_impl(name, exportable_module_cls, source_model_artifacts, export_config, exported_model_artifacts):
    # upstream body, plus lightweight_conversion
    model = source_model_artifacts.model
    module = exportable_module_cls(model)
    converter = E.converter_utils.Converter()
    for signature_name, (sample_inputs, _) in module.get_sample_inputs(source_model_artifacts.text_model_config, export_config).items():
        converter.add_signature(signature_name, module.eval(), sample_kwargs=sample_inputs)
    lrt_model = converter.convert(lightweight_conversion=export_config.experimental_lightweight_conversion, strict_export=False)
    model_path = os.path.join(export_config.work_dir, f"{name}.tflite")
    lrt_model.export(model_path)
    del lrt_model, converter
    gc.collect()
    recipe_list = export_config.quantization_recipe.split(",") if export_config.quantization_recipe else [None]
    for recipe in recipe_list:
        model_path = E.maybe_quantize_model(model_path, recipe)
        gc.collect()
    additional = exported_model_artifacts.additional_model_paths or {}
    additional[name] = model_path
    return dataclasses.replace(exported_model_artifacts, additional_model_paths=additional)


def _release(modules, label):
    n = 0
    for mod in modules:
        if mod is None:
            continue
        for p in mod.parameters():
            n += p.numel()
            p.data = torch.empty(0, dtype=p.dtype)
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass
    print(f"[skinnova] released {n / 1e9:.2f} B parameters after the {label}", flush=True)


_text_export = E.export_text_prefill_decode_model
_additional_export = E.export_additional_models


def _text_then_release(source_model_artifacts, export_config, exported_model_artifacts):
    out = _text_export(source_model_artifacts, export_config, exported_model_artifacts)
    m = source_model_artifacts.model.model
    _release((m.language_model.layers, getattr(m, "audio_tower", None), getattr(m, "embed_audio", None)), "text export")
    return out


def _additional_then_release(source_model_artifacts, export_config, exported_model_artifacts):
    out = _additional_export(source_model_artifacts, export_config, exported_model_artifacts)
    lm = source_model_artifacts.model.model.language_model
    _release([getattr(lm, k, None) for k in ("embed_tokens_per_layer", "per_layer_model_projection", "embed_tokens")],
             "per-layer embedder export")
    return out


E.export_additional_models_impl = _additional_impl
E.export_text_prefill_decode_model = _text_then_release
E.export_additional_models = _additional_then_release

if __name__ == "__main__":
    cli.main()
