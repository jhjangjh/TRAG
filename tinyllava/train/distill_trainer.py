# Modified by the TRAG authors from TinyLLaVA Factory. See NOTICE.
"""Teacher-student trainer for TRAG and the comparison distillation objectives.

The TRAG equations themselves live in :mod:`tinyllava.train.trag`. This module
extracts response-to-vision attention from FlashAttention models (via Q/K
hooks), assembles the complete objective, and implements the baselines.
"""
import math
import os
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers import Trainer
from transformers.modeling_utils import unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.trainer import _is_peft_model, has_length, logger

from .tinyllava_trainer import LLaVATrainer, LengthGroupedSampler, get_length_grouped_indices
from .trag import (
    aggregate_visual_attention,
    intermediate_layers,
    trag_loss as compute_trag_loss,
    update_entropy_bounds,
)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    bsz, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(bsz, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(bsz, num_kv_heads * n_rep, slen, head_dim)

def _gather_rope_cos_sin(cos, sin, position_ids, unsqueeze_dim: int):
    # cos/sin: [S,D] or [B,S,D]
    if position_ids is not None and cos.dim() == 2:
        cos = cos[position_ids]  # [B,S,D]
        sin = sin[position_ids]  # [B,S,D]
    if cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(unsqueeze_dim)  # [1,1,S,D]
        sin = sin.unsqueeze(0).unsqueeze(unsqueeze_dim)
    elif cos.dim() == 3:
        cos = cos.unsqueeze(unsqueeze_dim)  # [B,1,S,D]
        sin = sin.unsqueeze(unsqueeze_dim)
    return cos, sin

def apply_rotary_pos_emb_qwen(q, k, cos, sin, position_ids=None, unsqueeze_dim: int = 1):
    cos, sin = _gather_rope_cos_sin(cos, sin, position_ids, unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class QKRoPEAttentionExtractor:
    """
    FlashAttention friendly attention extraction:
      - Hook q/k projections (q_proj/k_proj or fused qkv_proj)
      - Hook rotary_emb forward to capture (cos,sin)
      - Compute attention weights from Q,K (+RoPE) without outputs.attentions
    """
    def __init__(self, model: torch.nn.Module, layer_indices=None):
        self.model = model
        self.layers = self._get_layers()
        cfg = self._get_text_cfg()

        self.num_layers = getattr(cfg, "num_hidden_layers", len(self.layers))
        self.layer_indices = layer_indices if layer_indices is not None else list(range(min(self.num_layers, len(self.layers))))

        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = getattr(cfg, "num_key_value_heads", self.num_heads)
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.head_dim = getattr(cfg, "head_dim", None)
        if self.head_dim is None:
            self.head_dim = cfg.hidden_size // self.num_heads

        self.q_cache = {}
        self.k_cache = {}
        self.pos_cache = {}
        self.registered_layers = []
        self._hooks = []

    def _unwrap_model(self, model):
        # Unwrap DeepSpeed/DDP wrapper if present
        if hasattr(model, "module"):
            return model.module
        return model

    def _get_text_cfg(self):
        model = self._unwrap_model(self.model)
        if hasattr(model, "config") and hasattr(model.config, "text_config"):
            return model.config.text_config
        if hasattr(model, "language_model") and hasattr(model.language_model, "config"):
            return model.language_model.config
        if hasattr(model, "config"):
            return model.config
        raise AttributeError("Cannot find text config")

    def _get_layers(self):
        model = self._unwrap_model(self.model)
        if hasattr(model, "language_model"):
            lm = model.language_model
            if hasattr(lm, "model") and hasattr(lm.model, "layers"):
                return lm.model.layers
            if hasattr(lm, "layers"):
                return lm.layers
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return model.model.layers
        raise AttributeError("Cannot find transformer layers. Please adjust _get_layers().")

    def _clear(self):
        self.q_cache.clear()
        self.k_cache.clear()
        self.pos_cache.clear()

    def _create_q_hook(self, layer_idx):
        def hook(m, inp, out):
            self.q_cache[layer_idx] = out
        return hook

    def _create_k_hook(self, layer_idx):
        def hook(m, inp, out):
            self.k_cache[layer_idx] = out
        return hook

    def _create_qkv_hook(self, layer_idx):
        def hook(m, inp, out):
            # out: [B,S,proj_dim]
            bsz, slen, proj_dim = out.shape
            q_dim = self.num_heads * self.head_dim
            k_dim = self.num_kv_heads * self.head_dim
            v_dim = self.num_kv_heads * self.head_dim

            if proj_dim >= (q_dim + k_dim + v_dim):
                self.q_cache[layer_idx] = out[:, :, :q_dim]
                self.k_cache[layer_idx] = out[:, :, q_dim:q_dim + k_dim]
            elif proj_dim % 3 == 0:
                one = proj_dim // 3
                self.q_cache[layer_idx] = out[:, :, :one]
                self.k_cache[layer_idx] = out[:, :, one:2*one]
        return hook

    def _create_attn_pre_hook(self, layer_idx):
        def hook(module, args, kwargs):
            position_ids = None
            if len(args) >= 3 and isinstance(args[2], torch.Tensor):
                position_ids = args[2]
            if "position_ids" in kwargs and isinstance(kwargs["position_ids"], torch.Tensor):
                position_ids = kwargs["position_ids"]
            if position_ids is not None:
                item = self.pos_cache.get(layer_idx, {})
                item["position_ids"] = position_ids.detach()
                self.pos_cache[layer_idx] = item
        return hook

    def _create_rotary_hook(self, layer_idx):
        def hook(module, inp, out):
            if isinstance(out, tuple) and len(out) == 2:
                cos, sin = out
                if isinstance(cos, torch.Tensor) and isinstance(sin, torch.Tensor):
                    item = self.pos_cache.get(layer_idx, {})
                    item["cos"] = cos.detach()
                    item["sin"] = sin.detach()
                    self.pos_cache[layer_idx] = item
        return hook

    def _register(self):
        self._hooks = []
        self.registered_layers = []

        for layer_idx in self.layer_indices:
            if layer_idx >= len(self.layers):
                continue
            layer = self.layers[layer_idx]
            attn = getattr(layer, "self_attn", None)
            if attn is None:
                continue

            attached = False

            if hasattr(attn, "q_proj") and hasattr(attn, "k_proj"):
                self._hooks.append(attn.q_proj.register_forward_hook(self._create_q_hook(layer_idx)))
                self._hooks.append(attn.k_proj.register_forward_hook(self._create_k_hook(layer_idx)))
                attached = True
            elif hasattr(attn, "qkv_proj"):
                self._hooks.append(attn.qkv_proj.register_forward_hook(self._create_qkv_hook(layer_idx)))
                attached = True
            elif hasattr(attn, "c_attn"):
                self._hooks.append(attn.c_attn.register_forward_hook(self._create_qkv_hook(layer_idx)))
                attached = True

            self._hooks.append(attn.register_forward_pre_hook(self._create_attn_pre_hook(layer_idx), with_kwargs=True))

            if hasattr(attn, "rotary_emb"):
                self._hooks.append(attn.rotary_emb.register_forward_hook(self._create_rotary_hook(layer_idx)))

            if attached:
                self.registered_layers.append(layer_idx)

        # DDP-safe: warn instead of raise
        if len(self.registered_layers) == 0:
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            if local_rank == 0:
                logger.warning("QKRoPEAttentionExtractor: no attention projection hooks registered. "
                               "Check attn module names (q_proj/k_proj or qkv_proj).")

    def _remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    class Ctx:
        def __init__(self, outer):
            self.outer = outer
        def __enter__(self):
            self.outer._clear()
            self.outer._register()
            return self
        def __exit__(self, exc_type, exc_val, exc_tb):
            self.outer._remove()
            return False

    def capture(self):
        return QKRoPEAttentionExtractor.Ctx(self)

    def compute_attn(
        self,
        layer_idx,
        q_indices,
        k_start,
        k_end,
        attention_mask=None,
        full_softmax_over_seq=True,
        batch_idx: Optional[int] = None,
    ):
        """
        Batch-safe attention extraction.

        If batch_idx is not None:
        - Slice cached q/k (and attention_mask, position_ids) to a single sample (bsz=1)
        - Return shape: [H, Q, K]

        If batch_idx is None:
        - Compute for full batch
        - Return shape: [B, H, Q, K]
        - (q_indices can be [Q] shared or [B,Q] per-sample)
        """

        def _warn_once(key: str, msg: str):
            if not hasattr(self, "_warned_keys"):
                self._warned_keys = set()
            if key in self._warned_keys:
                return
            self._warned_keys.add(key)
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            if local_rank == 0:
                logger.warning(f"QKRoPEAttentionExtractor: {msg}")

        if layer_idx not in self.q_cache or layer_idx not in self.k_cache:
            _warn_once(f"missing_qk_{layer_idx}", f"Missing Q/K cache at layer {layer_idx}. Skipping.")
            return None
        if layer_idx not in self.pos_cache or "cos" not in self.pos_cache[layer_idx] or "sin" not in self.pos_cache[layer_idx]:
            _warn_once(f"missing_rope_{layer_idx}", f"Missing RoPE cos/sin at layer {layer_idx}. Skipping.")
            return None

        q_proj = self.q_cache[layer_idx]  # [B,S,q_dim]
        k_proj = self.k_cache[layer_idx]  # [B,S,k_dim]
        cos = self.pos_cache[layer_idx]["cos"]
        sin = self.pos_cache[layer_idx]["sin"]
        position_ids = self.pos_cache[layer_idx].get("position_ids", None)

        device = q_proj.device
        bsz, seq_len, _ = q_proj.shape

        # ---- slice to single sample if requested ----
        if batch_idx is not None:
            if not (0 <= int(batch_idx) < bsz):
                _warn_once(f"bad_batch_idx_{layer_idx}", f"batch_idx out of range at layer {layer_idx}: {batch_idx} vs B={bsz}.")
                return None
            b = int(batch_idx)
            q_proj = q_proj[b:b+1]
            k_proj = k_proj[b:b+1]
            # LLaMA returns already-positioned [B,S,D] rotary tables; Qwen2
            # returns shared [S,D] tables. Slice only the former with Q/K.
            if cos.dim() == 3 and cos.shape[0] > 1:
                cos = cos[b:b+1]
                sin = sin[b:b+1]
            bsz = 1

            if position_ids is not None:
                # position_ids could be [B,S] or [S]
                if position_ids.dim() == 2 and position_ids.shape[0] > 1:
                    position_ids = position_ids[b:b+1]
            # attention_mask could be [S], [1,S], [B,S]
            if attention_mask is not None:
                am0 = attention_mask
                if torch.is_tensor(am0):
                    if am0.dim() == 2 and am0.shape[0] > 1:
                        attention_mask = am0[b:b+1]
                    elif am0.dim() == 1:
                        attention_mask = am0.unsqueeze(0)
                    else:
                        attention_mask = am0  # [1,S]
        else:
            # full batch mode: normalize attention_mask shape to [B,S] if provided
            if attention_mask is not None:
                am0 = attention_mask
                if torch.is_tensor(am0):
                    if am0.dim() == 1:
                        attention_mask = am0.unsqueeze(0).expand(bsz, -1)  # [B,S]
                    elif am0.dim() == 2 and am0.shape[0] == 1 and bsz > 1:
                        attention_mask = am0.expand(bsz, -1)               # [B,S]
                    else:
                        attention_mask = am0

        # ---- q_indices handling ----
        q_indices = q_indices.to(device)
        if q_indices.numel() == 0:
            return None

        # allow q_indices: [Q] (shared) or [B,Q] (per-sample) in full-batch mode
        if batch_idx is not None:
            if q_indices.dim() == 2:
                q_indices = q_indices[int(batch_idx)]
        else:
            if q_indices.dim() == 2 and q_indices.shape[0] != bsz:
                _warn_once(
                    f"qidx_shape_{layer_idx}",
                    f"q_indices batch mismatch at layer {layer_idx}: got {tuple(q_indices.shape)}, expected first dim B={bsz}.",
                )
                return None

        k_start = int(k_start)
        k_end = int(min(k_end, seq_len))
        if k_start >= k_end:
            return None

        # ---- Q,K reshape ----
        q = q_proj.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)      # [B,H,S,D]
        k = k_proj.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)   # [B,Hkv,S,D]

        # ---- RoPE ----
        q, k = apply_rotary_pos_emb_qwen(q, k, cos, sin, position_ids, unsqueeze_dim=1)

        # ---- expand kv heads if needed ----
        if self.num_kv_groups > 1:
            k = repeat_kv(k, self.num_kv_groups)  # [B,H,S,D]

        scale = self.head_dim ** -0.5

        # ---- gather q_slice ----
        if q_indices.dim() == 1:
            # shared indices: [Q]
            q_slice = q[:, :, q_indices, :]  # [B,H,Q,D]
            q_pos_for_causal = q_indices.view(1, 1, -1, 1)  # broadcastable
        else:
            # per-sample indices: [B,Q]
            # gather over S dim (dim=2)
            idx = q_indices[:, None, :, None].expand(bsz, self.num_heads, q_indices.shape[1], self.head_dim)  # [B,H,Q,D]
            q_slice = torch.gather(q, dim=2, index=idx)  # [B,H,Q,D]
            q_pos_for_causal = q_indices[:, None, :, None]  # [B,1,Q,1]

        if full_softmax_over_seq:
            scores_full = torch.matmul(q_slice.float(), k.float().transpose(-2, -1)) * scale  # [B,H,Q,S]

            # causal mask
            key_pos = torch.arange(seq_len, device=device).view(1, 1, 1, seq_len)  # [1,1,1,S]
            scores_full = scores_full.masked_fill(key_pos > q_pos_for_causal, torch.finfo(scores_full.dtype).min)

            # pad mask
            if attention_mask is not None:
                am0 = attention_mask.to(device)
                if am0.dim() == 1:
                    am0 = am0.unsqueeze(0)
                if am0.shape[1] != seq_len:
                    _warn_once(
                        f"am_len_mismatch_{layer_idx}",
                        f"attention_mask length mismatch at layer {layer_idx}: got {am0.shape[1]}, expected {seq_len}. Skipping.",
                    )
                    return None
                if am0.shape[0] == 1 and bsz > 1:
                    am0 = am0.expand(bsz, -1)

                if am0.shape[0] != bsz:
                    _warn_once(
                        f"am_bsz_mismatch_{layer_idx}",
                        f"attention_mask batch mismatch at layer {layer_idx}: got {am0.shape[0]}, expected {bsz}. Skipping.",
                    )
                    return None

                am = am0.view(bsz, 1, 1, seq_len)
                scores_full = scores_full.masked_fill(am == 0, torch.finfo(scores_full.dtype).min)

            probs = F.softmax(scores_full, dim=-1)  # [B,H,Q,S]
            out = probs[:, :, :, k_start:k_end]     # [B,H,Q,K]

        else:
            k_slice = k[:, :, k_start:k_end, :]  # [B,H,K,D]
            scores = torch.matmul(q_slice, k_slice.transpose(-2, -1)) * scale  # [B,H,Q,K]
            out = F.softmax(scores, dim=-1)  # [B,H,Q,K]

        if batch_idx is not None:
            return out[0]  # [H,Q,K]
        return out         # [B,H,Q,K]


def get_distill_modality_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."

    # Separate multimodal and language-only samples
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0]) if any(l > 0 for l in lengths) else ([], [])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0]) if any(l < 0 for l in lengths) else ([], [])

    # If only one modality exists, fall back to regular length grouping
    if len(mm_indices) == 0 or len(lang_indices) == 0:
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    # Unlike the SFT sampler, the two trailing partial megabatches are appended
    # separately at the end instead of being merged. Only those tail samples
    # can share a batch across modalities.
    if len(last_mm) > 0:
        megabatches.append(last_mm)
    if len(last_lang) > 0:
        megabatches.append(last_lang)

    return [i for megabatch in megabatches for i in megabatch]


class DistillLengthGroupedSampler(LengthGroupedSampler):
    """Modality-grouped sampler used for distillation (see above)."""

    def __iter__(self):
        if self.group_by_modality:
            return iter(get_distill_modality_grouped_indices(
                self.lengths, self.batch_size, self.world_size, generator=self.generator))
        return super().__iter__()


def _get_first_vision_span(b_idx: int, num_images_per_sample, vision_spans):
    """Return the first ``(start, end)`` vision span of sample ``b_idx`` or None.

    ``vision_spans[b_idx]`` is ``[]`` (no image) or a list of expanded-sequence
    ``(start, end)`` pairs, one per image; ``end`` is exclusive.
    """
    if num_images_per_sample is None or vision_spans is None:
        return None
    if b_idx >= len(num_images_per_sample) or num_images_per_sample[b_idx] < 1:
        return None
    if b_idx >= len(vision_spans):
        return None

    spans = vision_spans[b_idx]
    if not isinstance(spans, (list, tuple)) or len(spans) == 0:
        return None

    st, ed = spans[0]
    return int(st), int(ed)


class DistillLLaVATrainer(LLaVATrainer):
    """Knowledge-distillation trainer.

    ``distill_mode`` selects the objective:

    - ``kd``: LM loss + forward KL on response tokens.
    - ``llava-kd``: ``kd`` + KL on vision-token logits + vision affinity
      distillation (LLaVA-KD).
    - ``compodistill``: ``kd`` + visual-attention alignment (CompoDistill; the
      attention term is enabled in the DFT stage).
    - ``trag``: Eq. 15/24 of the paper, i.e. LM loss, response KL and
      LLaVA-KD vision terms (Eqs. 25-27) plus the TRAG term (Eqs. 8-14).
    """

    LOG_EVERY = 100
    ENTROPY_LOG_EVERY = 200

    def __init__(
        self,
        teacher_model,
        distill_mode="trag",
        train_stage="dpt",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.distill_mode = distill_mode
        if distill_mode == "trag" and self.args.gradient_checkpointing:
            # TRAG reads Q/K captured by forward hooks inside the decoder
            # layers. Reentrant checkpointing runs those layers under no_grad,
            # which would silently cut every TRAG gradient; the non-reentrant
            # variant keeps them (and still recomputes activations).
            self.args.gradient_checkpointing_kwargs = {
                **(self.args.gradient_checkpointing_kwargs or {}), "use_reentrant": False}
        self.use_attn_loss = False
        self.train_stage = train_stage

        self.loss_fct = nn.KLDivLoss(reduction="batchmean")
        self.niter = 0

        # Stop training after this many consecutive NaN/Inf gradient norms.
        self.consecutive_nan_grad_norm = 0
        self.max_consecutive_nan_grad_norm = 5

        # Frozen teacher on the training device and precision.
        if self.args.bf16:
            target_dtype = torch.bfloat16
        elif self.args.fp16:
            target_dtype = torch.float16
        else:
            target_dtype = torch.float32
        self.teacher_model = teacher_model.to(device=self.args.device, dtype=target_dtype)
        self.teacher_model.eval()
        self.teacher_model.requires_grad_(False)
        if self.args.n_gpu > 1 and not self.args.deepspeed:
            self.teacher_model = torch.nn.DataParallel(self.teacher_model)
        logger.info(f"Teacher model ready on {self.args.device} ({target_dtype}).")

        if self.distill_mode == "compodistill" and self.train_stage == "dft":
            self._init_compodistill_targets()
            self.use_attn_loss = True

        # Eq. (12): running mini-batch entropy min/max, beta=0.995.
        self.entropy_min_ema = None
        self.entropy_max_ema = None
        self.entropy_ema_momentum = 0.995
        # Before the EMA exists, bounds come from a warm-up buffer of teacher
        # entropies collected over the first 100 forward passes.
        self.entropy_warmup_steps = 100
        self.entropy_buffer = []
        self.entropy_buffer_max = 500
        self.entropy_fallback_min = 2.5
        self.entropy_fallback_max = 7.0

    # ------------------------------------------------------------------
    # Entropy statistics (Eq. 12)
    # ------------------------------------------------------------------
    def _get_tb_writer(self):
        for cb in self.callback_handler.callbacks:
            if hasattr(cb, "tb_writer") and cb.tb_writer is not None:
                return cb.tb_writer
        return None

    def _log_entropy_to_tensorboard(self, all_entropy, all_lambda):
        """Log teacher-entropy and forward-KL-weight distributions."""
        tb_writer = self._get_tb_writer()
        if tb_writer is None:
            return

        entropy_float = all_entropy.float().cpu()
        lambda_float = all_lambda.float().cpu()

        tb_writer.add_histogram("trag/entropy_dist", entropy_float, self.niter)
        tb_writer.add_histogram("trag/lambda_dist", lambda_float, self.niter)

        tb_writer.add_scalar("trag/lambda_mean", lambda_float.mean().item(), self.niter)
        tb_writer.add_scalar("trag/lambda_std", lambda_float.std().item(), self.niter)
        tb_writer.add_scalar("trag/entropy_mean", entropy_float.mean().item(), self.niter)
        tb_writer.add_scalar("trag/entropy_std", entropy_float.std().item(), self.niter)

        if self.entropy_min_ema is not None:
            tb_writer.add_scalar("trag/ema_min", self.entropy_min_ema, self.niter)
            tb_writer.add_scalar("trag/ema_max", self.entropy_max_ema, self.niter)

    def _update_entropy_stats_batch(self, all_entropy: torch.Tensor):
        """Update the bounds once per forward mini-batch, after the loss is built.

        Statistics are rank-local (no cross-rank synchronization) and are not
        checkpointed, so they restart on resume or at a new stage.
        """
        if all_entropy.numel() == 0:
            return
        with torch.no_grad():
            entropy = all_entropy.detach().float()
            if self.entropy_min_ema is None:
                # Collect warm-up values (keep the latest 500); keep collecting
                # past step 100 until at least 100 values are available.
                if self.niter < self.entropy_warmup_steps or len(self.entropy_buffer) < 100:
                    self.entropy_buffer.extend(entropy.cpu().flatten().tolist())
                    self.entropy_buffer = self.entropy_buffer[-self.entropy_buffer_max:]
            else:
                previous = (torch.tensor(self.entropy_min_ema),
                            torch.tensor(self.entropy_max_ema))
                lower, upper = update_entropy_bounds(
                    entropy, previous, momentum=self.entropy_ema_momentum)
                self.entropy_min_ema = lower.item()
                self.entropy_max_ema = upper.item()

    def _maybe_init_entropy_ema(self):
        """Initialize the EMA from the warm-up buffer's min/max (Eq. 12)."""
        if (self.niter >= self.entropy_warmup_steps
                and self.entropy_min_ema is None
                and len(self.entropy_buffer) >= 100):
            values = torch.tensor(self.entropy_buffer, dtype=torch.float32)
            self.entropy_min_ema = values.min().item()
            self.entropy_max_ema = values.max().item()
            self.entropy_buffer = []

    def _get_entropy_bounds(self, entropy_q: torch.Tensor):
        """Return (H_min, H_max) for Eq. 12.

        Uses the running EMA once initialized. During warm-up, samples with at
        least 10 response tokens use their own min/max; shorter samples use the
        fixed range [2.5, 7.0].
        """
        if self.entropy_min_ema is not None:
            lower, upper = self.entropy_min_ema, self.entropy_max_ema
        elif entropy_q.numel() >= 10:
            return entropy_q.detach().float().amin(), entropy_q.detach().float().amax()
        else:
            lower, upper = self.entropy_fallback_min, self.entropy_fallback_max
        return (torch.tensor(lower, device=entropy_q.device, dtype=torch.float32),
                torch.tensor(upper, device=entropy_q.device, dtype=torch.float32))

    # ------------------------------------------------------------------
    # CompoDistill setup
    # ------------------------------------------------------------------
    def _get_text_num_layers(self, m: nn.Module) -> int:
        cfg = getattr(m, "config", None)
        if cfg is None:
            raise AttributeError("Model has no config; cannot infer num_hidden_layers.")

        def _read_layers(layer_cfg):
            for name in ("num_hidden_layers", "n_layer"):
                value = getattr(layer_cfg, name, None)
                if value is not None:
                    return int(value)
            return None

        if hasattr(cfg, "text_config"):
            num_layers = _read_layers(cfg.text_config)
            if num_layers is not None:
                return num_layers

        num_layers = _read_layers(cfg)
        if num_layers is not None:
            return num_layers

        raise AttributeError("Cannot infer num_hidden_layers or n_layer from model.config.")

    # Fixed target layers per backbone depth, following the official
    # CompoDistill implementation (roughly 30-70% of the decoder depth).
    _COMPODISTILL_STUDENT_LAYERS = {24: list(range(8, 17)), 28: list(range(9, 20))}
    _COMPODISTILL_TEACHER_LAYERS = {
        24: list(range(8, 17)),
        28: list(range(9, 20)),
        32: list(range(10, 23)),
        36: list(range(11, 26)),
        40: list(range(12, 29)),
    }

    def _init_compodistill_targets(self):
        sL = self._get_text_num_layers(unwrap_model(self.model))
        tL = self._get_text_num_layers(unwrap_model(self.teacher_model))
        if sL not in self._COMPODISTILL_STUDENT_LAYERS:
            raise AssertionError(f"Not recommended student num_hidden_layers: {sL}")
        if tL not in self._COMPODISTILL_TEACHER_LAYERS:
            raise AssertionError(f"Not recommended teacher num_hidden_layers: {tL}")

        self.student_target_layers = self._COMPODISTILL_STUDENT_LAYERS[sL]
        self.teacher_target_layers = self._COMPODISTILL_TEACHER_LAYERS[tL]
        # Sliding-window group size n = m - k + 1.
        self.teacher_group = len(self.teacher_target_layers) - len(self.student_target_layers) + 1
        if self.teacher_group < 1:
            raise AssertionError(f"Invalid grouping: teacher shall be deeper (sL={sL}, tL={tL})")

    # ------------------------------------------------------------------
    # Trainer overrides
    # ------------------------------------------------------------------
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return DistillLengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size,
                lengths=lengths,
                group_by_modality=True,
            )
        return Trainer._get_train_sampler(self)

    def log(self, logs):
        """Log, and stop training after repeated NaN/Inf gradient norms."""
        if "grad_norm" in logs:
            grad_norm = logs["grad_norm"]
            if grad_norm is None or (isinstance(grad_norm, float) and not math.isfinite(grad_norm)):
                self.consecutive_nan_grad_norm += 1
                if self.args.local_rank == 0:
                    logger.warning(
                        f"NaN/Inf gradient norm detected "
                        f"({self.consecutive_nan_grad_norm}/{self.max_consecutive_nan_grad_norm})."
                    )
                    logs["nan_grad_norm_count"] = self.consecutive_nan_grad_norm

                if self.consecutive_nan_grad_norm >= self.max_consecutive_nan_grad_norm:
                    error_msg = (f"Training stopped: Gradient norm has been NaN/Inf for "
                                 f"{self.consecutive_nan_grad_norm} consecutive steps!")
                    if self.args.local_rank == 0:
                        logger.error(error_msg)
                        logs["training_stopped"] = 1
                        logs["stop_reason"] = "consecutive_nan_grad_norm"
                    super().log(logs)
                    raise RuntimeError(error_msg)
            else:
                if self.consecutive_nan_grad_norm > 0 and self.args.local_rank == 0:
                    logger.info(f"Gradient norm recovered (grad_norm={grad_norm:.4f}).")
                    logs["nan_grad_norm_recovered"] = 1
                self.consecutive_nan_grad_norm = 0

        super().log(logs)

    # ------------------------------------------------------------------
    # Shared loss components
    # ------------------------------------------------------------------
    def _pop_labels(self, inputs):
        if self.label_smoother is not None and "labels" in inputs:
            return inputs.pop("labels")
        return None

    def _check_student_outputs(self, model, outputs, labels, inputs):
        """Bookkeeping from ``transformers.Trainer.compute_loss``."""
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                self.label_smoother(outputs, labels, shift_labels=True)
            else:
                self.label_smoother(outputs, labels)
        elif isinstance(outputs, dict) and "loss" not in outputs:
            raise ValueError(
                "The model did not return a loss from the inputs, only the following keys: "
                f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
            )

    def _masked_response_logits(self, outputs, teacher_outputs, labels, teacher_labels):
        """Next-token logits at supervised response positions, or ``(None, None)``
        if teacher and student disagree on the number of response tokens."""
        logits = outputs["logits"]
        voc_size = logits.shape[-1]
        student_mask = labels[..., 1:].reshape(-1).to(logits.device) != -100
        teacher_mask = teacher_labels[..., 1:].reshape(-1).to(logits.device) != -100

        student = logits[..., :-1, :].reshape(-1, voc_size)[student_mask]
        teacher = teacher_outputs["logits"][..., :-1, :].reshape(-1, voc_size)[teacher_mask]
        if teacher.shape != student.shape:
            if self.args.local_rank == 0:
                logger.warning(f"Response-logit shape mismatch (teacher {tuple(teacher.shape)}, "
                               f"student {tuple(student.shape)}); skipping response KL.")
            return None, None
        return student, teacher

    def _vision_logit_terms(self, outputs, teacher_outputs, num_images, spans,
                            teacher_num_images, teacher_spans, kl_reduction):
        """Per-sample vision-token distillation terms (LLaVA-KD; Eq. 27).

        Returns two lists: KL on vision-token logits and 0.5 times the cosine
        distance between the vision-logit affinity matrices.
        """
        voc_size = outputs["logits"].shape[-1]
        kl_terms, affinity_terms = [], []
        for b_idx in range(len(num_images)):
            span_s = _get_first_vision_span(b_idx, num_images, spans)
            span_t = _get_first_vision_span(b_idx, teacher_num_images, teacher_spans)
            if span_s is None or span_t is None:
                continue

            st = max(span_s[0], span_t[0])
            ed = min(span_s[1], span_t[1])
            if st >= ed:
                continue
            if not (0 <= st < ed <= teacher_outputs["logits"].shape[1]):
                continue
            if not (0 <= st < ed <= outputs["logits"].shape[1]):
                continue

            student_logits = outputs["logits"][b_idx:b_idx + 1, st:ed, :]
            teacher_logits = teacher_outputs["logits"][b_idx:b_idx + 1, st:ed, :]

            kl_terms.append(F.kl_div(
                F.log_softmax(student_logits.view(-1, voc_size), dim=-1),
                F.softmax(teacher_logits.view(-1, voc_size), dim=-1),
                reduction=kl_reduction,
            ))

            student_affinity = torch.matmul(student_logits, student_logits.permute(0, 2, 1))
            teacher_affinity = torch.matmul(teacher_logits, teacher_logits.permute(0, 2, 1))
            similarity = F.cosine_similarity(
                teacher_affinity.view(-1).unsqueeze(0),
                student_affinity.view(-1).unsqueeze(0),
            )
            affinity_terms.append(0.5 * (1 - similarity.mean()))
        return kl_terms, affinity_terms

    def _log_losses(self, losses):
        if self.niter % self.LOG_EVERY == 0 and self.args.local_rank == 0:
            self.log({k: float(v.detach().cpu()) if torch.is_tensor(v) else v
                      for k, v in losses.items() if v is not None})

    # ------------------------------------------------------------------
    # Objectives
    # ------------------------------------------------------------------
    def compute_loss(self, model, inputs, return_outputs=False):
        objectives = {
            "kd": self._compute_kd_loss,
            "llava-kd": self._compute_kd_loss,
            "compodistill": self._compute_compodistill_loss,
            "trag": self._compute_trag_objective,
        }
        if self.distill_mode not in objectives:
            raise ValueError(f"Unknown distill_mode: {self.distill_mode}")
        self.niter += 1
        loss, outputs = objectives[self.distill_mode](model, inputs)
        return (loss, outputs) if return_outputs else loss

    def _compute_kd_loss(self, model, inputs):
        """``kd``: LM + response KL; ``llava-kd`` adds the mean vision terms."""
        labels = self._pop_labels(inputs)
        with torch.no_grad():
            (teacher_outputs, teacher_labels, _, _, _,
             teacher_num_images, teacher_spans) = self.teacher_model(**inputs)
        outputs, student_labels, _, _, _, num_images, spans = model(**inputs)
        self._check_student_outputs(model, outputs, labels, inputs)

        llm_loss = outputs["loss"]
        loss = llm_loss
        student, teacher = self._masked_response_logits(outputs, teacher_outputs, student_labels, teacher_labels)
        forward_distillation_loss = None
        if student is not None:
            forward_distillation_loss = self.loss_fct(F.log_softmax(student, dim=-1), F.softmax(teacher, dim=-1))
            loss = loss + forward_distillation_loss

        losses = {"train/llm_loss": llm_loss, "train/forward_distillation_loss": forward_distillation_loss}
        if self.distill_mode == "llava-kd":
            kl_terms, affinity_terms = self._vision_logit_terms(
                outputs, teacher_outputs, num_images, spans,
                teacher_num_images, teacher_spans, kl_reduction="batchmean")
            if kl_terms:
                forward_visual_distillation = torch.stack(kl_terms).mean()
                llm_visual_rela_distill_loss = torch.stack(affinity_terms).mean()
                loss = loss + forward_visual_distillation + llm_visual_rela_distill_loss
                losses["train/forward_visual_distillation"] = forward_visual_distillation
                losses["train/llm_visual_rela_distill_loss"] = llm_visual_rela_distill_loss

        self._log_losses(losses)
        return loss, outputs

    def _compute_compodistill_loss(self, model, inputs):
        """CompoDistill: LM + response KL + visual-attention alignment (DFT)."""
        labels = self._pop_labels(inputs)
        with torch.no_grad():
            (teacher_outputs, teacher_labels, _, _, _,
             teacher_num_images, teacher_spans) = self.teacher_model(**inputs, output_attentions=self.use_attn_loss)
        if labels is not None:
            inputs["labels"] = labels
        outputs, student_labels, *_ = model(**inputs, output_attentions=self.use_attn_loss)
        self._check_student_outputs(model, outputs, labels, inputs)

        loss = outputs["loss"]
        student, teacher = self._masked_response_logits(outputs, teacher_outputs, student_labels, teacher_labels)
        forward_distillation_loss = None
        if student is not None:
            forward_distillation_loss = self.loss_fct(
                F.log_softmax(student.float(), dim=-1), F.softmax(teacher.float(), dim=-1))
            loss = loss + forward_distillation_loss

        attn_loss = None
        if self.use_attn_loss:
            attn_loss = self._compodistill_attention_loss(
                outputs, teacher_outputs, teacher_num_images, teacher_spans, like=outputs["loss"])
            loss = loss + attn_loss

        self._log_losses({"train/forward_distillation_loss": forward_distillation_loss,
                          "train/compodistill_attn_loss": attn_loss})
        return loss, outputs

    def _compodistill_attention_loss(self, outputs, teacher_outputs, teacher_num_images, teacher_spans, like):
        """CompoDistill Eq. (2): LADL = 1 - (1/k) sum_j cos(mean_{G_j} A_t, A_s) on visual columns."""
        def attentions(o):
            return o.get("attentions", None) if isinstance(o, dict) else getattr(o, "attentions", None)

        teacher_attn = attentions(teacher_outputs)
        student_attn = attentions(outputs)
        if teacher_attn is None or student_attn is None:
            raise ValueError("output_attentions=True but attentions are missing from teacher/student outputs.")
        # Keep per-layer lists; stacking all [L,B,H,S,S] tensors would OOM.
        teacher_attn = list(teacher_attn)
        student_attn = list(student_attn)

        eps = 1e-8
        student_layers = self.student_target_layers
        k = len(student_layers)
        n = self.teacher_group
        # Teacher groups G_j = LT[j : j + n], j = 0..k-1 (sliding window).
        teacher_groups = [self.teacher_target_layers[j:j + n] for j in range(k)]
        if any(len(g) != n for g in teacher_groups):
            raise RuntimeError(f"[CompoDistill] Bad teacher_groups; expected window n={n} for all groups.")

        loss_sum = None
        valid_b = 0
        tea_seq = teacher_attn[0].shape[-1]
        stu_seq = student_attn[0].shape[-1]
        for b_idx in range(teacher_attn[0].shape[0]):
            span = _get_first_vision_span(b_idx, teacher_num_images, teacher_spans)
            if span is None:
                continue
            st, ed = span
            if not (0 <= st < ed <= tea_seq) or not (0 <= st < ed <= stu_seq):
                continue

            # Head-averaged attention to visual columns: [k, S, V].
            try:
                stu_sel = torch.stack(
                    [student_attn[l_s][b_idx, :, :, st:ed].mean(dim=0) for l_s in student_layers], dim=0)
            except IndexError:
                continue

            if any(l_t < 0 or l_t >= len(teacher_attn) for g in teacher_groups for l_t in g):
                continue
            tea_grouped = torch.stack([
                torch.stack([teacher_attn[l_t][b_idx, :, :, st:ed].mean(dim=0) for l_t in g], dim=0).mean(dim=0)
                for g in teacher_groups
            ], dim=0)

            tea_flat = tea_grouped.reshape(k, -1).float()
            stu_flat = stu_sel.reshape(k, -1).float()
            tea_flat = tea_flat / (tea_flat.norm(dim=1, keepdim=True) + eps)
            stu_flat = stu_flat / (stu_flat.norm(dim=1, keepdim=True) + eps)
            ladl_b = 1.0 - (tea_flat * stu_flat).sum(dim=1).mean()

            loss_sum = ladl_b if loss_sum is None else loss_sum + ladl_b
            valid_b += 1

        if valid_b == 0:
            return torch.zeros_like(like)
        return loss_sum / float(valid_b)

    def _compute_trag_objective(self, model, inputs):
        """Eq. 15/24: LM + response KL + vision terms (LLaVA-KD) + TRAG.

        Reductions follow LLaVA-KD (whose weights lambda_alpha=1 and
        lambda_beta=0.5 the paper retains): every term is a token mean, and
        per-sample terms are averaged over samples with an image.
        """
        # Intermediate layers with relative depth in [0.3, 0.6] (Sec. 5.2).
        student_layers = intermediate_layers(self._get_text_num_layers(unwrap_model(self.model)))
        teacher_layers = intermediate_layers(self._get_text_num_layers(unwrap_model(self.teacher_model)))
        if self.niter == 1 and self.args.local_rank == 0:
            logger.info(f"TRAG student_layers={student_layers}, teacher_layers={teacher_layers}")

        labels = self._pop_labels(inputs)
        teacher_extractor = QKRoPEAttentionExtractor(self.teacher_model, layer_indices=teacher_layers)
        student_extractor = QKRoPEAttentionExtractor(model, layer_indices=student_layers)
        with torch.no_grad():
            with teacher_extractor.capture():
                (teacher_outputs, teacher_labels, _, _, _,
                 teacher_num_images, teacher_spans) = self.teacher_model(**inputs)
        with student_extractor.capture():
            outputs, student_labels, student_mask, _, _, num_images, spans = model(**inputs)
        self._check_student_outputs(model, outputs, labels, inputs)

        # Eq. (25): HF token-mean cross entropy over response tokens.
        llm_loss = outputs["loss"]
        zero = llm_loss * 0  # Keeps the graph connected for absent terms.

        # Eq. (26): response forward KL, mean over response tokens.
        student, teacher = self._masked_response_logits(outputs, teacher_outputs, student_labels, teacher_labels)
        forward_distillation_loss = zero
        if student is not None:
            forward_distillation_loss = self.loss_fct(
                F.log_softmax(student.float(), dim=-1), F.softmax(teacher.float(), dim=-1))

        # Eq. (27): vision-logit KL (mean over visual tokens) and affinity
        # distillation, each averaged over samples with an image.
        kl_terms, affinity_terms = self._vision_logit_terms(
            outputs, teacher_outputs, num_images, spans,
            teacher_num_images, teacher_spans, kl_reduction="batchmean")
        forward_visual_distillation = torch.stack(kl_terms).mean() if kl_terms else zero
        llm_visual_rela_distill_loss = torch.stack(affinity_terms).mean() if affinity_terms else zero

        # Eqs. (8-14): TRAG on response-to-vision attention.
        trag_terms, entropies, weights = [], [], []
        samples = self._trag_attention_samples(
            student_labels, student_mask, teacher_extractor, student_extractor, teacher_layers,
            student_layers, num_images, spans, teacher_num_images, teacher_spans)
        for teacher_attention, student_attention in samples:
            sample_loss, entropy, weight = self._trag_sample_loss(teacher_attention, student_attention)
            trag_terms.append(sample_loss)
            entropies.append(entropy)
            weights.append(weight)
        trag_loss = torch.stack(trag_terms).mean() if trag_terms else zero

        loss = (llm_loss + forward_distillation_loss + forward_visual_distillation
                + llm_visual_rela_distill_loss + trag_loss)

        # The current loss uses the previous EMA bounds; statistics are
        # updated once after all sample losses are built.
        if entropies:
            all_entropy = torch.cat(entropies)
            self._update_entropy_stats_batch(all_entropy)
            self._maybe_init_entropy_ema()
            if self.niter % self.ENTROPY_LOG_EVERY == 0 and self.args.local_rank == 0:
                self._log_entropy_to_tensorboard(all_entropy, torch.cat(weights))

        self._log_losses({
            "train/llm_loss": llm_loss,
            "train/forward_distillation_loss": forward_distillation_loss,
            "train/forward_visual_distillation": forward_visual_distillation,
            "train/llm_visual_rela_distill_loss": llm_visual_rela_distill_loss,
            "train/trag_loss": trag_loss,
            "train/trag_valid_samples": len(trag_terms),
            "train/visdistill_valid_samples": len(kl_terms),
        })
        return loss, outputs

    def _trag_attention_samples(self, expanded_labels, expanded_mask, teacher_extractor, student_extractor,
                                teacher_layers, student_layers,
                                num_images, spans, teacher_num_images, teacher_spans):
        """Eqs. (8-9) per sample: head- and layer-averaged response-to-vision
        attention, renormalized on the visual span.

        ``expanded_labels`` / ``expanded_mask`` are the model's labels and
        attention mask after visual-token insertion, truncation and padding.
        Returns ``(teacher [Q,K], student [Q,K])`` for every sample with an
        image and at least one response token after it.
        """
        eps = 1e-8
        if not (student_extractor.registered_layers and teacher_extractor.registered_layers):
            return []

        seq_len_s = student_extractor.q_cache[student_extractor.registered_layers[0]].shape[1]
        seq_len_t = teacher_extractor.q_cache[teacher_extractor.registered_layers[0]].shape[1]
        if seq_len_s != seq_len_t or expanded_labels.shape[1] != seq_len_s:
            if self.args.local_rank == 0:
                logger.warning(f"TRAG: sequence length mismatch (teacher {seq_len_t}, student {seq_len_s}, "
                               f"labels {expanded_labels.shape[1]}); skipping TRAG this step.")
            return []
        if expanded_mask is None:
            expanded_mask = torch.ones_like(expanded_labels)

        samples = []
        for b_idx in range(expanded_labels.shape[0]):
            span_s = _get_first_vision_span(b_idx, num_images, spans)
            span_t = _get_first_vision_span(b_idx, teacher_num_images, teacher_spans)
            if span_s is None or span_t is None:
                continue
            st = max(span_s[0], span_t[0])
            ed = min(span_s[1], span_t[1])
            if st >= ed:
                continue

            # Response-token query positions (supervised labels after the image).
            q_indices = torch.nonzero(expanded_labels[b_idx] != -100).squeeze(-1)
            q_indices = q_indices[q_indices >= ed]
            if q_indices.numel() == 0:
                continue
            am_exp = expanded_mask[b_idx]

            def layer_attention(extractor, layer):
                if layer not in extractor.registered_layers:
                    return None
                return extractor.compute_attn(
                    layer_idx=layer, q_indices=q_indices, k_start=st, k_end=ed,
                    attention_mask=am_exp.unsqueeze(0), full_softmax_over_seq=True, batch_idx=b_idx,
                )

            def head_mean(extractor, layers):
                # [H,Q,K] -> [Q,K] for every layer that produced attention.
                maps = (layer_attention(extractor, layer) for layer in layers)
                return [attn.float().mean(dim=0) for attn in maps if attn is not None]

            teacher_layer_attn = head_mean(teacher_extractor, teacher_layers)
            student_layer_attn = head_mean(student_extractor, student_layers)
            if not teacher_layer_attn or not student_layer_attn:
                continue
            samples.append((aggregate_visual_attention(torch.stack(teacher_layer_attn), eps),
                            aggregate_visual_attention(torch.stack(student_layer_attn), eps)))
        return samples

    def _trag_sample_loss(self, teacher_attention, student_attention):
        """Eqs. (11-14) for one sample, averaged over its response tokens.

        Returns ``(loss, entropy [Q], forward-KL weight lambda [Q])``.
        """
        eps = 1e-8
        entropy = -(teacher_attention * teacher_attention.clamp_min(eps).log()).sum(dim=-1)
        entropy_min, entropy_max = self._get_entropy_bounds(entropy)
        out = compute_trag_loss(
            teacher_attention=teacher_attention,
            student_attention=student_attention,
            entropy_min=entropy_min,
            entropy_max=entropy_max,
            eps=eps,
            reduction="mean",
        )
        return out.loss, out.entropy.detach(), out.forward_weight.detach()
