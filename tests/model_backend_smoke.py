from __future__ import annotations

import argparse
import json

import numpy as np

from repsteer.config import ModelConfig
from repsteer.models.huggingface import HuggingFaceBackend
from repsteer.types import Intervention


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    args = parser.parse_args()
    backend = HuggingFaceBackend(ModelConfig(
        model_id=args.model_id, dtype="float32", device_map="cpu", use_chat_template=True))
    texts = ["The cat rested beside the window.", "A dog waited beside the garden gate."]
    layer = min(1, backend.n_layers - 1)
    representations = backend.sentence_representations(
        texts, layer=layer, pooling="mean", batch_size=2, max_length=64)
    if representations.shape[0] != 2 or representations.ndim != 2:
        raise RuntimeError(f"Unexpected representation shape: {representations.shape}")
    if not np.isfinite(representations).all():
        raise RuntimeError("Representations contain non-finite values")
    vector = representations[1] - representations[0]
    norm = np.linalg.norm(vector)
    if norm <= 0 or not np.isfinite(norm):
        raise RuntimeError(f"Invalid steering vector norm: {norm}")
    intervention = Intervention(layer, vector / norm, 1.0, "all")
    generation_args = dict(max_length=128, max_new_tokens=6, temperature=0.0,
                           top_p=1.0, do_sample=False, seed=42)
    prompt = "Continue in one short sentence: The animal looked outside."
    baseline = backend.generate([prompt], intervention=None, **generation_args)[0]
    steered = backend.generate([prompt], intervention=intervention, **generation_args)[0]
    captured_text, states = backend.generate_with_representations(
        prompt, capture_layers=[layer], intervention=intervention, **generation_args)
    nll = backend.sequence_nll(texts, max_length=64, batch_size=2, intervention=intervention)
    choice_probs = backend.next_token_choice_probabilities(
        ["Choose the first letter. Answer:"], ["A", "B"], max_length=64,
        batch_size=1, intervention=intervention)
    if not baseline.strip() or not steered.strip() or not captured_text.strip():
        raise RuntimeError("Model-backed generation returned an empty continuation")
    if states[layer].shape != (representations.shape[1],) or not np.isfinite(states[layer]).all():
        raise RuntimeError("Online hidden-state capture is invalid")
    if any(tokens <= 0 or not np.isfinite(loss) for loss, tokens in nll):
        raise RuntimeError("Intervention-aware NLL output is invalid")
    if choice_probs.shape != (1, 2) or not np.isclose(choice_probs.sum(), 1.0):
        raise RuntimeError("Next-token choice probabilities are invalid")
    print(json.dumps({
        "model_id": args.model_id, "n_layers": backend.n_layers, "layer": layer,
        "representation_shape": list(representations.shape),
        "vector_norm_before_normalization": float(norm), "baseline": baseline,
        "steered": steered, "captured": captured_text,
        "nll": nll, "choice_probabilities": choice_probs.tolist()},
        ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
