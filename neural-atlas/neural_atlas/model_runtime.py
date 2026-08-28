from __future__ import annotations

import asyncio
import os
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any, AsyncIterator, Callable

import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from .safetensors_index import SafetensorsIndex


def _config_value(config: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        value = getattr(config, name, None)
        if value is not None:
            return value
    return default


def _choose_device(requested: str) -> torch.device:
    requested = requested.lower().strip()
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _choose_dtype(requested: str, device: torch.device) -> torch.dtype:
    requested = requested.lower().strip()
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if requested != "auto":
        try:
            return mapping[requested]
        except KeyError as exc:
            raise ValueError("NEURAL_ATLAS_DTYPE must be auto, float32, float16, or bfloat16") from exc
    if device.type == "cuda":
        return torch.float16
    return torch.float32


def _tensor_summary(vector: torch.Tensor, top_k: int = 8) -> dict[str, Any]:
    values = vector.detach().float().reshape(-1)
    if values.numel() == 0:
        return {
            "norm": 0.0,
            "mean": 0.0,
            "standardDeviation": 0.0,
            "minimum": 0.0,
            "maximum": 0.0,
            "topDimensions": [],
        }
    count = min(top_k, values.numel())
    magnitudes, indices = torch.topk(values.abs(), k=count)
    return {
        "norm": float(torch.linalg.vector_norm(values).item()),
        "mean": float(values.mean().item()),
        "standardDeviation": float(values.std(unbiased=False).item()),
        "minimum": float(values.min().item()),
        "maximum": float(values.max().item()),
        "topDimensions": [
            {
                "index": int(index),
                "value": float(values[index].item()),
                "magnitude": float(magnitude),
            }
            for magnitude, index in zip(magnitudes.tolist(), indices.tolist())
        ],
    }


class ModelRuntime:
    """Lazy model loader and token-by-token measured inference tracer."""

    def __init__(
        self,
        model_id: str | None = None,
        revision: str | None = None,
        device: str | None = None,
        dtype: str | None = None,
    ) -> None:
        self.model_id = model_id or os.getenv(
            "NEURAL_ATLAS_MODEL_ID", "EleutherAI/pythia-70m-deduped"
        )
        self.revision = revision or os.getenv("NEURAL_ATLAS_REVISION", "main")
        self.requested_device = device or os.getenv("NEURAL_ATLAS_DEVICE", "auto")
        self.requested_dtype = dtype or os.getenv("NEURAL_ATLAS_DTYPE", "auto")
        self.device = _choose_device(self.requested_device)
        self.dtype = _choose_dtype(self.requested_dtype, self.device)

        self.model: Any | None = None
        self.tokenizer: Any | None = None
        self.checkpoint_path: Path | None = None
        self.index: SafetensorsIndex | None = None
        self.layers: list[Any] = []
        self.final_norm: Any | None = None
        self.output_head: Any | None = None
        self.architecture_family = "unloaded"
        self._mlp_output_modules: list[Any | None] = []
        self._router_modules: list[Any | None] = []
        self._parameter_names: dict[int, str] = {}
        self._load_lock = asyncio.Lock()
        self._inference_lock = asyncio.Lock()
        self._activation_store: OrderedDict[str, dict[int, dict[int, dict[str, Any]]]] = OrderedDict()

    @property
    def loaded(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def status(self) -> dict[str, Any]:
        return {
            "loaded": self.loaded,
            "modelId": self.model_id,
            "revision": self.revision,
            "device": str(self.device),
            "dtype": str(self.dtype).replace("torch.", ""),
            "checkpointPath": str(self.checkpoint_path) if self.checkpoint_path else None,
            "indexed": self.index is not None,
            "architectureFamily": self.architecture_family,
        }

    async def load(self, model_id: str | None = None) -> dict[str, Any]:
        async with self._load_lock:
            if model_id and model_id != self.model_id:
                self.model_id = model_id
                self.model = None
                self.tokenizer = None
                self.index = None
                self.checkpoint_path = None
            if not self.loaded:
                await asyncio.to_thread(self._load_sync)
            return self.architecture()

    def _resolve_checkpoint(self) -> Path:
        candidate = Path(self.model_id).expanduser()
        if candidate.exists():
            return candidate.resolve()
        location = snapshot_download(
            repo_id=self.model_id,
            revision=self.revision,
            allow_patterns=[
                "*.safetensors",
                "*.json",
                "*.model",
                "*.tiktoken",
                "*.txt",
            ],
        )
        return Path(location).resolve()

    def _load_sync(self) -> None:
        checkpoint = self._resolve_checkpoint()
        tokenizer = AutoTokenizer.from_pretrained(
            str(checkpoint),
            local_files_only=True,
            use_fast=True,
        )
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        model_kwargs = {
            "local_files_only": True,
            "torch_dtype": self.dtype,
        }
        try:
            model = AutoModelForCausalLM.from_pretrained(
                str(checkpoint),
                attn_implementation="eager",
                **model_kwargs,
            )
        except (TypeError, ValueError):
            model = AutoModelForCausalLM.from_pretrained(str(checkpoint), **model_kwargs)
        model.eval()
        model.to(self.device)

        self.model = model
        self.tokenizer = tokenizer
        self.checkpoint_path = checkpoint
        self.index = SafetensorsIndex.from_path(checkpoint)
        self._parameter_names = {id(parameter): name for name, parameter in model.named_parameters()}
        self._resolve_architecture()

    async def ensure_index(self) -> SafetensorsIndex:
        if self.index is not None:
            return self.index
        async with self._load_lock:
            if self.index is None:
                if self.checkpoint_path is None:
                    self.checkpoint_path = await asyncio.to_thread(self._resolve_checkpoint)
                self.index = await asyncio.to_thread(SafetensorsIndex.from_path, self.checkpoint_path)
        return self.index

    def _resolve_architecture(self) -> None:
        assert self.model is not None
        model = self.model
        if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
            self.architecture_family = "gpt_neox"
            self.layers = list(model.gpt_neox.layers)
            self.final_norm = getattr(model.gpt_neox, "final_layer_norm", None)
            self.output_head = getattr(model, "embed_out", None) or getattr(model, "lm_head", None)
        elif hasattr(model, "model") and hasattr(model.model, "layers"):
            self.architecture_family = "decoder_layers"
            self.layers = list(model.model.layers)
            self.final_norm = getattr(model.model, "norm", None)
            self.output_head = getattr(model, "lm_head", None)
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            self.architecture_family = "gpt2"
            self.layers = list(model.transformer.h)
            self.final_norm = getattr(model.transformer, "ln_f", None)
            self.output_head = getattr(model, "lm_head", None)
        else:
            raise RuntimeError(
                f"Unsupported model architecture {type(model).__name__}; add a ModelRuntime adapter"
            )

        self._mlp_output_modules = [self._find_mlp_output(layer) for layer in self.layers]
        self._router_modules = [self._find_router(layer) for layer in self.layers]

    @staticmethod
    def _find_mlp_output(layer: Any) -> Any | None:
        mlp = getattr(layer, "mlp", None) or getattr(layer, "feed_forward", None)
        if mlp is None:
            return None
        for name in ("dense_4h_to_h", "down_proj", "c_proj", "dense_2"):
            module = getattr(mlp, name, None)
            if module is not None and hasattr(module, "weight"):
                return module
        return None

    @staticmethod
    def _find_router(layer: Any) -> Any | None:
        mlp = getattr(layer, "mlp", None) or getattr(layer, "block_sparse_moe", None)
        if mlp is None:
            return None
        for name in ("router", "gate"):
            module = getattr(mlp, name, None)
            if module is not None and callable(module):
                return module
        return None

    def architecture(self) -> dict[str, Any]:
        if not self.loaded:
            return self.status()
        assert self.model is not None
        config = self.model.config
        parameter_count = sum(parameter.numel() for parameter in self.model.parameters())
        trainable_count = sum(
            parameter.numel() for parameter in self.model.parameters() if parameter.requires_grad
        )
        return {
            **self.status(),
            "modelClass": type(self.model).__name__,
            "modelType": getattr(config, "model_type", type(config).__name__),
            "parameterCount": parameter_count,
            "trainableParameterCount": trainable_count,
            "layerCount": len(self.layers),
            "hiddenSize": int(
                _config_value(config, "hidden_size", "n_embd", "d_model", default=0)
            ),
            "intermediateSize": int(
                _config_value(config, "intermediate_size", "n_inner", default=0) or 0
            ),
            "attentionHeads": int(
                _config_value(config, "num_attention_heads", "n_head", default=0)
            ),
            "keyValueHeads": int(
                _config_value(
                    config,
                    "num_key_value_heads",
                    default=_config_value(config, "num_attention_heads", "n_head", default=0),
                )
            ),
            "vocabularySize": int(_config_value(config, "vocab_size", default=0)),
            "maximumContext": int(
                _config_value(
                    config,
                    "max_position_embeddings",
                    "n_positions",
                    "seq_length",
                    default=0,
                )
            ),
            "supportsMlpContributions": any(
                module is not None for module in self._mlp_output_modules
            ),
            "supportsRouterCapture": any(module is not None for module in self._router_modules),
        }

    def _token_record(self, token_id: int) -> dict[str, Any]:
        assert self.tokenizer is not None
        piece = self.tokenizer.convert_ids_to_tokens(int(token_id))
        text = self.tokenizer.decode(
            [int(token_id)],
            clean_up_tokenization_spaces=False,
            skip_special_tokens=False,
        )
        return {"id": int(token_id), "piece": str(piece), "text": text}

    def _decode_top_logits(self, logits: torch.Tensor, top_k: int = 10) -> list[dict[str, Any]]:
        probabilities = torch.softmax(logits.float(), dim=-1)
        count = min(top_k, probabilities.numel())
        values, ids = torch.topk(probabilities, k=count)
        return [
            {
                **self._token_record(int(token_id)),
                "probability": float(probability),
                "logit": float(logits[int(token_id)].float().item()),
            }
            for probability, token_id in zip(values.tolist(), ids.tolist())
        ]

    def _sample_token(
        self,
        logits: torch.Tensor,
        *,
        temperature: float,
        top_p: float,
    ) -> int:
        if temperature <= 0:
            return int(torch.argmax(logits).item())
        scaled = logits.float() / max(float(temperature), 1e-5)
        probabilities = torch.softmax(scaled, dim=-1)
        top_p = min(1.0, max(0.01, float(top_p)))
        if top_p < 1.0:
            sorted_probabilities, sorted_indices = torch.sort(probabilities, descending=True)
            cumulative = torch.cumsum(sorted_probabilities, dim=-1)
            remove = cumulative - sorted_probabilities > top_p
            sorted_probabilities[remove] = 0
            sorted_probabilities /= sorted_probabilities.sum()
            sampled = torch.multinomial(sorted_probabilities, num_samples=1)
            return int(sorted_indices[sampled].item())
        return int(torch.multinomial(probabilities, num_samples=1).item())

    def _logit_lens(self, hidden: torch.Tensor, top_k: int = 3) -> list[dict[str, Any]]:
        if self.final_norm is None or self.output_head is None:
            return []
        with torch.inference_mode():
            projected = self.final_norm(hidden.reshape(1, -1))
            logits = self.output_head(projected)[0].float()
        return self._decode_top_logits(logits, top_k=top_k)

    def _attention_summary(
        self,
        attention: torch.Tensor | None,
        token_ids: torch.Tensor,
        edge_count: int = 8,
    ) -> dict[str, Any]:
        if attention is None:
            return {"headCount": 0, "meanEntropy": None, "edges": []}
        values = attention.detach().float()
        if values.ndim != 4:
            return {"headCount": 0, "meanEntropy": None, "edges": []}
        last_query = values[0, :, -1, :]
        probabilities = last_query / last_query.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        entropy = -(probabilities.clamp_min(1e-12).log() * probabilities).sum(dim=-1)
        peak_values, peak_positions = torch.max(last_query, dim=-1)
        count = min(edge_count, peak_values.numel())
        selected_values, selected_heads = torch.topk(peak_values, k=count)
        edges: list[dict[str, Any]] = []
        for weight, head in zip(selected_values.tolist(), selected_heads.tolist()):
            position = int(peak_positions[int(head)].item())
            token_id = int(token_ids[0, position].item())
            edges.append(
                {
                    "head": int(head),
                    "fromPosition": int(token_ids.shape[1] - 1),
                    "toPosition": position,
                    "weight": float(weight),
                    "toToken": self._token_record(token_id),
                }
            )
        return {
            "headCount": int(last_query.shape[0]),
            "meanEntropy": float(entropy.mean().item()),
            "edges": edges,
        }

    @staticmethod
    def _last_token_vector(value: Any) -> torch.Tensor | None:
        if isinstance(value, (tuple, list)):
            value = value[0] if value else None
        if not isinstance(value, torch.Tensor):
            return None
        if value.ndim >= 3:
            return value[0, -1]
        if value.ndim == 2:
            return value[-1]
        return value.reshape(-1)

    def _capture_hooks(
        self,
        mlp_capture: dict[int, dict[str, Any]],
        router_capture: dict[int, dict[str, Any]],
    ) -> list[Any]:
        handles: list[Any] = []
        for layer_index, module in enumerate(self._mlp_output_modules):
            if module is None:
                continue

            def capture_mlp(
                _module: Any,
                arguments: tuple[Any, ...],
                *,
                index: int = layer_index,
            ) -> None:
                vector = self._last_token_vector(arguments[0] if arguments else None)
                if vector is None:
                    return
                detached = vector.detach().float().cpu()
                summary = _tensor_summary(detached, top_k=16)
                mlp_capture[index] = {
                    "vector": detached.numpy().copy(),
                    "summary": summary,
                }

            handles.append(module.register_forward_pre_hook(capture_mlp))

        for layer_index, module in enumerate(self._router_modules):
            if module is None:
                continue

            def capture_router(
                _module: Any,
                _arguments: tuple[Any, ...],
                output: Any,
                *,
                index: int = layer_index,
            ) -> None:
                vector = self._last_token_vector(output)
                if vector is None or vector.numel() < 2:
                    return
                probabilities = torch.softmax(vector.detach().float(), dim=-1)
                count = min(8, probabilities.numel())
                values, ids = torch.topk(probabilities, k=count)
                router_capture[index] = {
                    "selected": [
                        {"expert": int(expert), "weight": float(weight)}
                        for weight, expert in zip(values.tolist(), ids.tolist())
                    ]
                }

            handles.append(module.register_forward_hook(capture_router))
        return handles

    def _remember_activations(
        self,
        session_id: str,
        generation_step: int,
        mlp_capture: dict[int, dict[str, Any]],
    ) -> None:
        session = self._activation_store.setdefault(session_id, {})
        session[generation_step] = {}
        for layer_index, capture in mlp_capture.items():
            module = self._mlp_output_modules[layer_index]
            if module is None:
                continue
            weight_name = self._parameter_names.get(id(module.weight))
            bias_name = (
                self._parameter_names.get(id(module.bias))
                if getattr(module, "bias", None) is not None
                else None
            )
            session[generation_step][layer_index] = {
                "activation": capture["vector"],
                "weightName": weight_name,
                "biasName": bias_name,
            }
        self._activation_store.move_to_end(session_id)
        while len(self._activation_store) > 8:
            self._activation_store.popitem(last=False)
        if len(session) > 128:
            oldest = sorted(session)[: len(session) - 128]
            for step in oldest:
                session.pop(step, None)

    def _forward_step(
        self,
        *,
        session_id: str,
        generation_step: int,
        token_ids: torch.Tensor,
        temperature: float,
        top_p: float,
        capture_attention: bool,
        capture_logit_lens: bool,
    ) -> dict[str, Any]:
        assert self.model is not None
        mlp_capture: dict[int, dict[str, Any]] = {}
        router_capture: dict[int, dict[str, Any]] = {}
        handles = self._capture_hooks(mlp_capture, router_capture)
        started = time.perf_counter()
        try:
            with torch.inference_mode():
                outputs = self.model(
                    input_ids=token_ids,
                    output_hidden_states=True,
                    output_attentions=capture_attention,
                    use_cache=False,
                    return_dict=True,
                )
        finally:
            for handle in handles:
                handle.remove()
        latency_ms = (time.perf_counter() - started) * 1000

        logits = outputs.logits[0, -1].float()
        next_token_id = self._sample_token(logits, temperature=temperature, top_p=top_p)
        hidden_states = list(outputs.hidden_states or [])
        attentions = list(outputs.attentions or []) if capture_attention else []
        layer_records: list[dict[str, Any]] = []
        for layer_index in range(len(self.layers)):
            hidden_index = min(layer_index + 1, len(hidden_states) - 1)
            current = hidden_states[hidden_index][0, -1].detach().float()
            previous = hidden_states[max(0, hidden_index - 1)][0, -1].detach().float()
            summary = _tensor_summary(current)
            summary["deltaNorm"] = float(torch.linalg.vector_norm(current - previous).item())
            summary["layer"] = layer_index
            summary["mlp"] = mlp_capture.get(layer_index, {}).get("summary")
            summary["router"] = router_capture.get(layer_index)
            summary["attention"] = self._attention_summary(
                attentions[layer_index] if layer_index < len(attentions) else None,
                token_ids,
            )
            summary["logitLens"] = (
                self._logit_lens(current, top_k=3) if capture_logit_lens else []
            )
            layer_records.append(summary)

        self._remember_activations(session_id, generation_step, mlp_capture)
        return {
            "type": "token_step",
            "sessionId": session_id,
            "generationStep": generation_step,
            "contextLength": int(token_ids.shape[1]),
            "currentInputToken": self._token_record(int(token_ids[0, -1].item())),
            "selectedToken": self._token_record(next_token_id),
            "topPredictions": self._decode_top_logits(logits, top_k=10),
            "layers": layer_records,
            "latencyMilliseconds": latency_ms,
        }

    async def stream_trace(
        self,
        *,
        prompt: str,
        max_new_tokens: int = 16,
        temperature: float = 0.0,
        top_p: float = 0.95,
        capture_attention: bool = True,
        capture_logit_lens: bool = True,
        seed: int = 0,
        session_id: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        await self.load()
        assert self.tokenizer is not None
        session_id = session_id or uuid.uuid4().hex
        encoded = self.tokenizer(prompt, return_tensors="pt")
        token_ids = encoded["input_ids"]
        maximum_context = int(self.architecture().get("maximumContext") or 2048)
        if token_ids.shape[1] > maximum_context - max_new_tokens:
            raise ValueError(
                f"Prompt has {token_ids.shape[1]} tokens; model context leaves insufficient generation space"
            )
        token_ids = token_ids.to(self.device)
        input_records = [self._token_record(int(value)) for value in token_ids[0].tolist()]
        yield {
            "type": "session_start",
            "sessionId": session_id,
            "prompt": prompt,
            "inputTokens": input_records,
            "architecture": self.architecture(),
            "measurement": {
                "attention": capture_attention,
                "logitLens": capture_logit_lens,
                "mlpActivations": True,
                "exactMlpContributions": True,
            },
        }

        generated: list[int] = []
        torch.manual_seed(int(seed))
        async with self._inference_lock:
            for generation_step in range(min(128, max(1, int(max_new_tokens)))):
                event = await asyncio.to_thread(
                    self._forward_step,
                    session_id=session_id,
                    generation_step=generation_step,
                    token_ids=token_ids,
                    temperature=float(temperature),
                    top_p=float(top_p),
                    capture_attention=bool(capture_attention),
                    capture_logit_lens=bool(capture_logit_lens),
                )
                next_token_id = int(event["selectedToken"]["id"])
                generated.append(next_token_id)
                yield event
                addition = torch.tensor([[next_token_id]], device=self.device, dtype=token_ids.dtype)
                token_ids = torch.cat((token_ids, addition), dim=1)
                if next_token_id == self.tokenizer.eos_token_id:
                    break

        yield {
            "type": "complete",
            "sessionId": session_id,
            "generatedTokenIds": generated,
            "generatedText": self.tokenizer.decode(
                generated,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ),
        }

    async def contribution(
        self,
        *,
        session_id: str,
        generation_step: int,
        layer: int,
        output_index: int,
        top_k: int = 32,
    ) -> dict[str, Any]:
        await self.load()
        return await asyncio.to_thread(
            self._contribution_sync,
            session_id,
            int(generation_step),
            int(layer),
            int(output_index),
            min(128, max(1, int(top_k))),
        )

    def _contribution_sync(
        self,
        session_id: str,
        generation_step: int,
        layer: int,
        output_index: int,
        top_k: int,
    ) -> dict[str, Any]:
        if layer < 0 or layer >= len(self._mlp_output_modules):
            raise IndexError(f"Layer {layer} is outside 0..{len(self._mlp_output_modules) - 1}")
        capture = (
            self._activation_store.get(session_id, {})
            .get(generation_step, {})
            .get(layer)
        )
        if capture is None:
            raise KeyError("The requested trace activation is no longer available")
        module = self._mlp_output_modules[layer]
        if module is None or not hasattr(module, "weight"):
            raise RuntimeError(f"Layer {layer} has no supported MLP output projection")
        weight = module.weight.detach().float().cpu()
        activation = torch.from_numpy(capture["activation"]).float()
        if weight.ndim != 2 or activation.numel() != weight.shape[1]:
            raise RuntimeError("Captured activation does not match the MLP output weight shape")
        if output_index < 0 or output_index >= weight.shape[0]:
            raise IndexError(f"Output index {output_index} is outside 0..{weight.shape[0] - 1}")
        row = weight[output_index]
        products = row * activation
        count = min(top_k, products.numel())
        magnitudes, input_indices = torch.topk(products.abs(), k=count)
        bias = (
            float(module.bias.detach().float().cpu()[output_index].item())
            if getattr(module, "bias", None) is not None
            else 0.0
        )
        weighted_sum = float(products.sum().item())
        entries: list[dict[str, Any]] = []
        weight_name = capture.get("weightName")
        for magnitude, input_index in zip(magnitudes.tolist(), input_indices.tolist()):
            model_weight = float(row[input_index].item())
            checkpoint_address = None
            if self.index is not None and weight_name in self.index.names():
                checkpoint_address = self.index.read_scalar(
                    weight_name, [output_index, int(input_index)]
                )
            entries.append(
                {
                    "inputIndex": int(input_index),
                    "activation": float(activation[input_index].item()),
                    "weight": model_weight,
                    "contribution": float(products[input_index].item()),
                    "absoluteContribution": float(magnitude),
                    "checkpointAddress": checkpoint_address,
                }
            )
        return {
            "sessionId": session_id,
            "generationStep": generation_step,
            "layer": layer,
            "module": "mlp_output_projection",
            "weightTensor": weight_name,
            "weightShape": list(weight.shape),
            "outputIndex": output_index,
            "inputWidth": int(weight.shape[1]),
            "weightedSum": weighted_sum,
            "bias": bias,
            "reconstructedLinearOutput": weighted_sum + bias,
            "topContributions": entries,
            "equation": "output[i] = bias[i] + sum_j(weight[i,j] * activation[j])",
            "exact": True,
        }
