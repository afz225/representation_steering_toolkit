from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Sequence

import numpy as np

from ..config import ModelConfig
from ..types import Intervention, Pooling


class HuggingFaceBackend:
    """Generic decoder-only Hugging Face backend with residual-stream hooks."""

    _LAYER_PATHS = (
        "model.layers",
        "model.model.layers",
        "model.language_model.layers",
        "transformer.h",
        "gpt_neox.layers",
        "language_model.model.layers",
        "language_model.layers",
    )

    def __init__(self, config: ModelConfig):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install repsteer[hf] to use HuggingFaceBackend") from exc

        self.torch = torch
        dtype = getattr(torch, config.dtype) if config.dtype != "auto" else "auto"
        tokenizer_id = config.tokenizer_id or config.model_id
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_id, trust_remote_code=config.trust_remote_code
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_id,
            torch_dtype=dtype,
            device_map=config.device_map,
            trust_remote_code=config.trust_remote_code,
        )
        self.model.eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.use_chat_template = config.use_chat_template
        self.layers = self._resolve_layers(config.layer_path)

    def _resolve_path(self, path: str):
        current = self.model
        for part in path.split("."):
            current = getattr(current, part)
        return current

    def _resolve_layers(self, explicit_path: str | None):
        paths = (explicit_path,) if explicit_path else self._LAYER_PATHS
        for path in paths:
            if path is None:
                continue
            try:
                layers = self._resolve_path(path)
            except AttributeError:
                continue
            if hasattr(layers, "__len__") and len(layers):
                return layers
        raise ValueError("Could not locate decoder layers; set model.layer_path in the config")

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    @property
    def input_device(self):
        return self.model.get_input_embeddings().weight.device

    def _check_layer(self, layer: int) -> None:
        if not 0 <= layer < self.n_layers:
            raise IndexError(f"Layer {layer} is outside [0, {self.n_layers - 1}]")

    def _pool(self, hidden, mask, pooling: Pooling):
        torch = self.torch
        mask = mask.bool()
        if pooling == "last":
            positions = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
            indices = positions.masked_fill(~mask, -1).amax(dim=1)
            return hidden[torch.arange(hidden.shape[0], device=hidden.device), indices]
        expanded = mask.unsqueeze(-1)
        if pooling == "mean":
            return (hidden * expanded).sum(dim=1) / expanded.sum(dim=1).clamp_min(1)
        fill = torch.inf if pooling == "min" else -torch.inf
        masked = hidden.masked_fill(~expanded, fill)
        return masked.amin(dim=1) if pooling == "min" else masked.amax(dim=1)

    def sentence_representations(
        self,
        texts: Sequence[str],
        layer: int,
        pooling: Pooling,
        batch_size: int = 8,
        max_length: int = 512,
    ) -> np.ndarray:
        self._check_layer(layer)
        arrays: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(self.input_device)
            with self.torch.inference_mode():
                output = self.model(**encoded, output_hidden_states=True, use_cache=False)
            # hidden_states[0] is the embedding output; layer k is at k + 1.
            pooled = self._pool(output.hidden_states[layer + 1], encoded.attention_mask, pooling)
            arrays.append(pooled.float().cpu().numpy())
        return np.concatenate(arrays, axis=0)

    def _format_prompts(self, prompts: Sequence[str]) -> list[str]:
        if not self.use_chat_template or not hasattr(self.tokenizer, "apply_chat_template"):
            return list(prompts)
        return [
            self._apply_chat_template(prompt)
            for prompt in prompts
        ]

    def _apply_chat_template(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

    @contextmanager
    def _steering_hook(self, intervention: Intervention | None) -> Iterator[None]:
        if intervention is None:
            yield
            return
        self._check_layer(intervention.layer)
        torch = self.torch

        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            delta = torch.as_tensor(intervention.vector, device=hidden.device, dtype=hidden.dtype)
            delta = intervention.coefficient * delta.view(1, 1, -1)
            modified = hidden.clone()
            if intervention.positions == "all":
                modified = modified + delta
            elif intervention.positions == "last":
                modified[:, -1:, :] = modified[:, -1:, :] + delta
            elif intervention.positions == "generated":
                # Cached decoding passes one token; prompt prefill passes the full sequence.
                if modified.shape[1] == 1:
                    modified = modified + delta
            else:
                raise ValueError(f"Unsupported position mode: {intervention.positions}")
            if isinstance(output, tuple):
                return (modified, *output[1:])
            return modified

        handle = self.layers[intervention.layer].register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()

    def generate(
        self,
        prompts: Sequence[str],
        *,
        max_length: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        do_sample: bool,
        seed: int,
        intervention: Intervention | None = None,
    ) -> list[str]:
        formatted = self._format_prompts(prompts)
        encoded = self.tokenizer(
            formatted, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
        ).to(self.input_device)
        self.torch.manual_seed(seed)
        kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if do_sample:
            kwargs.update(temperature=temperature, top_p=top_p)
        with self._steering_hook(intervention), self.torch.inference_mode():
            generated = self.model.generate(**encoded, **kwargs)
        prompt_width = encoded.input_ids.shape[1]
        return self.tokenizer.batch_decode(generated[:, prompt_width:], skip_special_tokens=True)


    def sequence_nll(
        self,
        texts: Sequence[str],
        *,
        max_length: int,
        batch_size: int = 1,
        intervention: Intervention | None = None,
    ) -> list[tuple[float, int]]:
        """Return summed causal NLL and scored-token count for each text."""

        results: list[tuple[float, int]] = []
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                add_special_tokens=True,
                return_tensors="pt",
            ).to(self.input_device)
            with self._steering_hook(intervention), self.torch.inference_mode():
                logits = self.model(**encoded, use_cache=False).logits
            shifted_logits = logits[:, :-1, :].float()
            shifted_labels = encoded.input_ids[:, 1:]
            # Exclude the first real token when left padding leaves it without causal context.
            shifted_mask = (encoded.attention_mask[:, 1:].bool() & encoded.attention_mask[:, :-1].bool())
            losses = self.torch.nn.functional.cross_entropy(
                shifted_logits.reshape(-1, shifted_logits.shape[-1]),
                shifted_labels.reshape(-1),
                reduction="none",
            ).reshape(shifted_labels.shape)
            losses = losses * shifted_mask
            for row in range(len(batch)):
                count = int(shifted_mask[row].sum().item())
                results.append((float(losses[row].sum().item()), count))
        return results

    def next_token_choice_probabilities(
        self,
        prompts: Sequence[str],
        choices: Sequence[str],
        *,
        max_length: int,
        batch_size: int = 8,
        intervention: Intervention | None = None,
    ) -> np.ndarray:
        """Score single-token answer labels at the next-token position."""

        choice_ids: list[int] = []
        for choice in choices:
            token_ids = self.tokenizer(
                f" {choice}", add_special_tokens=False
            ).input_ids
            if len(token_ids) != 1:
                raise ValueError(
                    f"Choice label {choice!r} is not one token with a leading space: {token_ids}"
                )
            choice_ids.append(int(token_ids[0]))

        arrays: list[np.ndarray] = []
        formatted = self._format_prompts(prompts)
        for start in range(0, len(formatted), batch_size):
            batch = list(formatted[start : start + batch_size])
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(self.input_device)
            with self._steering_hook(intervention), self.torch.inference_mode():
                logits = self.model(**encoded, use_cache=False).logits
            positions = self.torch.arange(
                encoded.attention_mask.shape[1], device=encoded.attention_mask.device
            ).expand_as(encoded.attention_mask)
            last = positions.masked_fill(~encoded.attention_mask.bool(), -1).amax(dim=1)
            batch_indices = self.torch.arange(logits.shape[0], device=logits.device)
            selected = logits[batch_indices, last, :][:, choice_ids].float()
            arrays.append(self.torch.softmax(selected, dim=-1).cpu().numpy())
        return np.concatenate(arrays, axis=0)

    def generate_with_representations(
        self,
        prompt: str,
        *,
        capture_layers: Sequence[int],
        max_length: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        do_sample: bool,
        seed: int,
        intervention: Intervention | None = None,
    ) -> tuple[str, dict[int, np.ndarray]]:
        """Generate and average each layer's last-position state over decoding calls."""

        for layer in capture_layers:
            self._check_layer(layer)
        formatted = self._format_prompts([prompt])[0]
        encoded = self.tokenizer(
            formatted,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(self.input_device)
        sums: dict[int, object] = {}
        counts: dict[int, int] = {}
        handles = []

        def capture_hook(layer: int):
            def hook(_module, _inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                value = hidden[:, -1, :].detach().float().sum(dim=0).cpu()
                sums[layer] = value if layer not in sums else sums[layer] + value
                counts[layer] = counts.get(layer, 0) + hidden.shape[0]

            return hook

        self.torch.manual_seed(seed)
        kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if do_sample:
            kwargs.update(temperature=temperature, top_p=top_p)

        with self._steering_hook(intervention):
            try:
                for layer in capture_layers:
                    handles.append(
                        self.layers[layer].register_forward_hook(capture_hook(layer))
                    )
                with self.torch.inference_mode():
                    generated = self.model.generate(**encoded, **kwargs)
            finally:
                for handle in handles:
                    handle.remove()

        prompt_width = encoded.input_ids.shape[1]
        text = self.tokenizer.decode(
            generated[0, prompt_width:], skip_special_tokens=True
        )
        missing = set(capture_layers) - set(sums)
        if missing:
            raise RuntimeError(f"Capture hooks were not called for layers: {sorted(missing)}")
        states = {
            layer: (sums[layer] / counts[layer]).numpy()
            for layer in capture_layers
        }
        return text, states
