from __future__ import annotations

import random
from contextlib import contextmanager
from typing import Iterator, Sequence

import numpy as np

from ..config import ModelConfig
from ..text import safe_text
from ..types import Intervention, Pooling
from .registry import ModelProfile, ModelRegistry, apply_profile, registry_for, select_modules_by_regex


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

    def __init__(self, config: ModelConfig, registry: ModelRegistry | str | None = None):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install repsteer[hf] to use HuggingFaceBackend") from exc

        self.torch = torch
        profile = self._match_profile(config, registry)
        config = apply_profile(config, profile)
        self.profile = profile
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
        # Extraction pads with the tokenizer's own default side (production never
        # overrode it); generation pads left.
        self.extraction_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        self.use_chat_template = config.use_chat_template
        self.layers = self._resolve_layers(config.layer_path, config.layer_regex)

    @staticmethod
    def _match_profile(config: ModelConfig, registry: ModelRegistry | str | None) -> ModelProfile | None:
        try:
            registry_obj = registry_for(config, registry)
        except (OSError, ValueError) as exc:
            raise ValueError(f"Could not load model registry: {exc}") from exc
        if registry_obj is None:
            return None
        return registry_obj.resolve(config.model_id)

    def _resolve_path(self, path: str):
        current = self.model
        for part in path.split("."):
            current = getattr(current, part)
        return current

    def _resolve_layers(self, explicit_path: str | None, layer_regex: str | None = None):
        if layer_regex and not explicit_path:
            self.layers = select_modules_by_regex(self.model, layer_regex)
            return self.layers
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

    def _pool(self, hidden, attention_mask, pooling: Pooling):
        """Pool one hooked layer output exactly as the production scripts did.

        ``mean`` is ``caa_model_search.SearchCaaModel.sentence_reps`` (cast to
        float32, then masked mean); ``hs_mean`` is
        ``geometry_olmo_reduced_full.PromptRepStore`` (masked mean accumulated in
        the model dtype, then cast). ``last`` indexes ``attention_mask.sum - 1``
        and ``min``/``max`` fill padding with the dtype's finite extremes, as
        production ``hs_last``/``hs_min``/``hs_max`` did.
        """
        torch = self.torch
        if pooling == "mean":
            hidden = hidden.float()
            mask = attention_mask.to(hidden.device).float().unsqueeze(-1)
            return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        if pooling == "last":
            last_idx = (attention_mask.sum(dim=1) - 1).long()
            selector = torch.arange(attention_mask.shape[0], device=attention_mask.device)
            return hidden[selector.to(hidden.device), last_idx.to(hidden.device)].float()
        mask = attention_mask.to(dtype=torch.bool).to(hidden.device).unsqueeze(-1)
        if pooling == "hs_mean":
            denom = mask.sum(dim=1).clamp(min=1).to(hidden.dtype)
            return ((hidden * mask.to(hidden.dtype)).sum(dim=1) / denom).float()
        if pooling == "min":
            return hidden.masked_fill(~mask, torch.finfo(hidden.dtype).max).min(dim=1).values.float()
        if pooling == "max":
            return hidden.masked_fill(~mask, torch.finfo(hidden.dtype).min).max(dim=1).values.float()
        raise ValueError(f"Unknown pooling {pooling!r}")

    def sentence_representations(
        self,
        texts: Sequence[str],
        layer: int,
        pooling: Pooling,
        batch_size: int = 8,
        max_length: int = 512,
    ) -> np.ndarray:
        """Pooled output of the resolved decoder block ``self.layers[layer]``.

        Texts are whitespace-normalized, no chat template is applied, and the
        tokenizer pads with its original default side.
        """
        self._check_layer(layer)
        torch = self.torch
        module = self.layers[layer]
        arrays: list[np.ndarray] = []
        generation_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = self.extraction_padding_side
        try:
            for start in range(0, len(texts), batch_size):
                batch = [safe_text(text) for text in texts[start : start + batch_size]]
                encoded = self.tokenizer(
                    batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
                encoded = {key: value.to(self.input_device) for key, value in encoded.items()}
                store = {}

                def hook(_module, _inputs, output):
                    store["hidden"] = (output[0] if isinstance(output, tuple) else output).detach()

                handle = module.register_forward_hook(hook)
                try:
                    with torch.no_grad():
                        self.model(**encoded)
                finally:
                    handle.remove()
                pooled = self._pool(store["hidden"], encoded["attention_mask"], pooling)
                arrays.append(pooled.cpu().numpy().astype(np.float32))
        finally:
            self.tokenizer.padding_side = generation_side
        return np.concatenate(arrays, axis=0).astype(np.float32)

    def _format_prompts(self, prompts: Sequence[str]) -> list[str]:
        if not self.use_chat_template or not hasattr(self.tokenizer, "apply_chat_template"):
            return list(prompts)
        return [
            self._apply_chat_template(prompt)
            for prompt in prompts
        ]

    def _apply_chat_template(self, prompt: str) -> str:
        """``SearchCaaModel.format_generation_prompt`` / ``StringJudge.format_judge_prompt``:
        try ``enable_thinking=False``, retry without it on TypeError, and fall back to
        the raw prompt if the template fails otherwise."""

        messages = [{"role": "user", "content": prompt}]
        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            try:
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                return prompt
        except Exception:
            return prompt

    @contextmanager
    def _steering_hook(self, intervention: Intervention | None) -> Iterator[None]:
        """``caa_model_search.SearchCaaModel.register_caa_hook`` arithmetic.

        The float32 vector is cast to the hidden dtype *before* scaling by
        ``float(coefficient)``; ``generated`` skips only the first multi-token
        (prefill) pass of the registration, as production did.
        """

        if intervention is None:
            yield
            return
        self._check_layer(intervention.layer)
        torch = self.torch
        vector = torch.from_numpy(np.asarray(intervention.vector, dtype=np.float32).copy())
        coefficient = float(intervention.coefficient)
        positions = intervention.positions
        if positions not in ("all", "last", "generated"):
            raise ValueError(f"Unsupported position mode: {positions}")
        state = {"prefill_seen": False}

        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            hidden = hidden.clone()
            delta = coefficient * vector.to(device=hidden.device, dtype=hidden.dtype).view(1, 1, -1)
            if positions == "generated":
                if hidden.shape[1] > 1 and not state["prefill_seen"]:
                    state["prefill_seen"] = True
                else:
                    hidden[:, :, :] += delta
            elif positions == "last":
                hidden[:, -1:, :] += delta
            else:
                hidden[:, :, :] += delta
            if isinstance(output, tuple):
                return (hidden,) + tuple(output[1:])
            return hidden

        handle = self.layers[intervention.layer].register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()

    def _seed_everything(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        self.torch.manual_seed(seed)

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
        normalize_whitespace: bool = False,
        retry_empty: bool = False,
        pad_with_eos: bool = False,
    ) -> list[str]:
        """Decode each prompt; with the production switches this is ``SearchCaaModel.generate``.

        ``normalize_whitespace`` applies ``safe_text`` to the prompt before the chat
        template and to the decoded continuation; ``retry_empty`` retries an empty
        continuation with ``(seed + 1, do_sample)`` then ``(seed, greedy)`` under the
        same hook registration. A single prompt is tokenized without padding.
        """

        if normalize_whitespace:
            prompts = [safe_text(prompt) for prompt in prompts]
        formatted = self._format_prompts(prompts)
        pad_token_id = self.tokenizer.eos_token_id if pad_with_eos else self.tokenizer.pad_token_id
        attempts = [(seed, do_sample)]
        if retry_empty:
            attempts += [(seed + 1, do_sample), (seed, False)]
        batches = [[text] for text in formatted] if retry_empty else [formatted]
        outputs: list[str] = []
        for batch in batches:
            if len(batch) == 1:
                encoded = self.tokenizer(
                    batch[0], truncation=True, max_length=max_length, return_tensors="pt"
                ).to(self.input_device)
            else:
                encoded = self.tokenizer(
                    batch, padding=True, truncation=True, max_length=max_length,
                    return_tensors="pt",
                ).to(self.input_device)
            prompt_width = encoded["input_ids"].shape[1]
            texts: list[str] = []
            # One hook registration per prompt; retries share it (production state).
            with self._steering_hook(intervention), self.torch.inference_mode():
                for attempt_seed, sample in attempts:
                    self._seed_everything(attempt_seed)
                    kwargs = {
                        "max_new_tokens": max_new_tokens,
                        "do_sample": sample,
                        "pad_token_id": pad_token_id,
                    }
                    if sample:
                        kwargs.update(temperature=temperature, top_p=top_p)
                    generated = self.model.generate(**encoded, **kwargs)
                    texts = self.tokenizer.batch_decode(
                        generated[:, prompt_width:], skip_special_tokens=True
                    )
                    if normalize_whitespace:
                        texts = [safe_text(text) for text in texts]
                    if all(texts):
                        break
            outputs.extend(texts)
        return outputs


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
