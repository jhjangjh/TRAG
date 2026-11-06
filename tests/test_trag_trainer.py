"""Tests for the entropy-bound schedule and the complete TRAG objective."""
from types import SimpleNamespace
from contextlib import nullcontext

import torch
import pytest
from transformers import Trainer

from tinyllava.train.distill_trainer import DistillLLaVATrainer
from tinyllava.train.tinyllava_trainer import LLaVATrainer


def bare_trainer():
    trainer = object.__new__(DistillLLaVATrainer)
    trainer.distill_mode = "trag"
    trainer.entropy_min_ema = trainer.entropy_max_ema = None
    trainer.entropy_ema_momentum = 0.995
    trainer.entropy_warmup_steps = 100
    trainer.entropy_buffer = []
    trainer.entropy_buffer_max = 500
    trainer.entropy_fallback_min = 2.5
    trainer.entropy_fallback_max = 7.0
    trainer.niter = 0
    return trainer


def test_warmup_buffer_and_min_max_initialization():
    trainer = bare_trainer()
    trainer.niter = 1
    trainer._update_entropy_stats_batch(torch.linspace(1.0, 3.0, 600))
    trainer._maybe_init_entropy_ema()
    assert trainer.entropy_min_ema is None
    assert len(trainer.entropy_buffer) == 500
    expected_min = trainer.entropy_buffer[0]
    trainer.niter = 100
    trainer._maybe_init_entropy_ema()
    assert trainer.entropy_min_ema == expected_min
    assert trainer.entropy_max_ema == 3.0
    assert trainer.entropy_buffer == []
    previous = trainer.entropy_min_ema
    trainer._update_entropy_stats_batch(torch.tensor([2.0, 5.0]))
    torch.testing.assert_close(torch.tensor(trainer.entropy_min_ema),
                               torch.tensor(0.995 * previous + 0.005 * 2.0))
    torch.testing.assert_close(torch.tensor(trainer.entropy_max_ema), torch.tensor(3.01))


def test_small_sample_fallback_and_sample_extrema():
    trainer = bare_trainer()
    lower, upper = trainer._get_entropy_bounds(torch.ones(9))
    assert lower == 2.5 and upper == 7.0
    entropy = torch.linspace(0.1, 0.9, 10)
    lower, upper = trainer._get_entropy_bounds(entropy)
    torch.testing.assert_close(lower, entropy.min())
    torch.testing.assert_close(upper, entropy.max())


def test_warmup_boundary_does_not_add_current_batch_when_buffer_is_ready():
    trainer = bare_trainer()
    trainer.niter = 99
    trainer._update_entropy_stats_batch(torch.linspace(1.0, 3.0, 100))
    trainer.niter = 100
    trainer._update_entropy_stats_batch(torch.tensor([-10.0, 20.0]))
    trainer._maybe_init_entropy_ema()
    assert trainer.entropy_min_ema == 1.0 and trainer.entropy_max_ema == 3.0


def test_sparse_warmup_can_finish_collecting_after_boundary():
    trainer = bare_trainer()
    trainer.niter = 100
    trainer.entropy_buffer = [1.0] * 99
    trainer._update_entropy_stats_batch(torch.tensor([3.0]))
    trainer._maybe_init_entropy_ema()
    assert trainer.entropy_min_ema == 1.0 and trainer.entropy_max_ema == 3.0


def test_rank_local_entropy_does_not_call_collectives(monkeypatch):
    import torch.distributed as dist
    monkeypatch.setattr(dist, "is_initialized", lambda: True)

    def forbidden(*args, **kwargs):
        raise AssertionError("Entropy statistics must not use collectives")

    monkeypatch.setattr(dist, "all_reduce", forbidden)
    trainer = bare_trainer()
    trainer.entropy_min_ema, trainer.entropy_max_ema = 0.2, 0.9
    trainer._update_entropy_stats_batch(torch.empty(0))
    assert trainer.entropy_min_ema == 0.2 and trainer.entropy_max_ema == 0.9
    trainer._update_entropy_stats_batch(torch.tensor([0.3, 0.8]))
    torch.testing.assert_close(torch.tensor(trainer.entropy_min_ema), torch.tensor(0.2005))
    torch.testing.assert_close(torch.tensor(trainer.entropy_max_ema), torch.tensor(0.8995))


def test_entropy_state_is_not_checkpointed():
    assert DistillLLaVATrainer._save_checkpoint is Trainer._save_checkpoint
    trainer = bare_trainer()
    assert trainer.entropy_min_ema is trainer.entropy_max_ema is None
    assert trainer.entropy_buffer == [] and trainer.niter == 0


def test_sft_keeps_model_token_mean_loss():
    trainer = object.__new__(LLaVATrainer)
    trainer.label_smoother = None
    trainer.args = SimpleNamespace(past_index=-1)
    labels = torch.tensor([[-100, 1, -100, -100], [-100, 1, 1, 1]])
    model = lambda **kwargs: SimpleNamespace(loss=torch.tensor(2.0))
    loss = trainer.compute_loss(model, {"labels": labels})
    torch.testing.assert_close(loss, torch.tensor(2.0))


def test_actual_compute_loss_matches_independent_arithmetic(monkeypatch):
    from tinyllava.train import distill_trainer as module
    import torch.nn.functional as F

    expanded_labels = torch.tensor([[-100, -100, -100, 1, -100, -100],
                                    [-100, -100, -100, 1, 2, 1]])

    class Model(torch.nn.Module):
        def __init__(self, offset):
            super().__init__()
            self.config = SimpleNamespace(num_hidden_layers=10)
            self.logits = torch.nn.Parameter(torch.arange(36).reshape(2, 6, 3).float().sin() + offset)
            self.raw = torch.nn.Parameter(torch.tensor([[[0.8, 0.1], [0.3, 0.2]],
                                                       [[0.01, 0.09], [0.2, 0.1]]]) + offset)

        def forward(self, **kwargs):
            return ({'loss': self.logits.square().mean(), 'logits': self.logits},
                    expanded_labels, None, None, None, [1, 1], [[(0, 2)], [(0, 2)]])

    class Extractor:
        def __init__(self, model, layer_indices):
            self.model = model
            self.registered_layers = layer_indices
            self.q_cache = {l: torch.zeros(2, 6, 1) for l in layer_indices}

        def capture(self):
            return nullcontext()

        def compute_attn(self, layer_idx, q_indices, batch_idx, **kwargs):
            row = self.model.raw[(layer_idx - 2) % 2, batch_idx]
            return row[None, None, :].expand(1, len(q_indices), 2)

    monkeypatch.setattr(module, 'QKRoPEAttentionExtractor', Extractor)
    trainer = bare_trainer()
    trainer.model = Model(0.1)
    trainer.teacher_model = Model(0.0).requires_grad_(False)
    trainer.args = SimpleNamespace(local_rank=-1, past_index=-1)
    trainer.label_smoother = None
    trainer.loss_fct = torch.nn.KLDivLoss(reduction="batchmean")
    trainer.train_stage = 'dft'
    trainer.niter = 10
    trainer.entropy_min_ema, trainer.entropy_max_ema = 0.1, 0.9
    inputs = {'input_ids': torch.tensor([[-200, 0, 1, 0, 0], [-200, 0, 1, 2, 1]]),
              'labels': expanded_labels[:, 1:].clone(), 'attention_mask': torch.ones(2, 5)}
    actual = trainer.compute_loss(trainer.model, inputs)

    s, t = trainer.model.logits, trainer.teacher_model.logits
    mask = expanded_labels[:, 1:] != -100
    # Token means, then means over samples with an image (LLaVA-KD reductions).
    expected = s.square().mean()  # model-provided token-mean CE
    expected = expected + F.kl_div(s[:, :-1][mask].log_softmax(-1), t[:, :-1][mask].softmax(-1),
                                   reduction='batchmean')
    expected += sum(F.kl_div(s[b, :2].log_softmax(-1), t[b, :2].softmax(-1), reduction='batchmean')
                    for b in range(2)) / 2
    affinities_s = s[:, :2] @ s[:, :2].transpose(1, 2)
    affinities_t = t[:, :2] @ t[:, :2].transpose(1, 2)
    expected += 0.5 * (1 - F.cosine_similarity(affinities_s.flatten(1), affinities_t.flatten(1))).mean()
    # Selected zero-based layers 3,4,5,6 (M=10): raw layer 1,0,1,0.
    pt = trainer.teacher_model.raw.detach().mean(0)
    ps = trainer.model.raw.mean(0)
    pt, ps = pt / (pt.sum(-1, keepdim=True) + 1e-8), ps / (ps.sum(-1, keepdim=True) + 1e-8)
    entropy = -(pt * pt.log()).sum(-1)
    # Current loss must use PREVIOUS EMA, not bounds from the current batch.
    weight = ((entropy - 0.1) / (0.9 - 0.1 + 1e-8)).clamp(0, 1)
    fwd = (pt * (pt.log() - ps.log())).sum(-1)
    rev = (ps * (ps.log() - pt.log())).sum(-1)
    # Each sample's token losses are identical rows, so its token mean is that row.
    expected += (weight * fwd + (1 - weight) * rev).mean()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.tensor(trainer.entropy_min_ema),
                               0.995 * torch.tensor(0.1) + 0.005 * entropy.min())
    torch.testing.assert_close(torch.tensor(trainer.entropy_max_ema),
                               0.995 * torch.tensor(0.9) + 0.005 * entropy.max())
    actual.backward()
    assert torch.isfinite(trainer.model.raw.grad).all()
    assert trainer.model.raw.grad.abs().sum() > 0


@pytest.mark.parametrize('backbone', ['qwen2', 'llama'])
def test_real_qk_rope_extraction_matches_eager_attention(backbone):
    from transformers import Qwen2Config, Qwen2ForCausalLM, LlamaConfig, LlamaForCausalLM
    from tinyllava.train.distill_trainer import QKRoPEAttentionExtractor
    config_cls, model_cls = ((Qwen2Config, Qwen2ForCausalLM) if backbone == 'qwen2'
                             else (LlamaConfig, LlamaForCausalLM))
    config = config_cls(vocab_size=16, hidden_size=16, intermediate_size=32,
                        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                        attention_dropout=0.0)
    config._attn_implementation = 'eager'
    model = model_cls(config).eval()
    ids = torch.tensor([[1, 2, 3, 4, 5], [2, 3, 4, 5, 0]])
    mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]])
    positions = torch.tensor([[0, 1, 2, 3, 4], [0, 0, 1, 2, 3]])
    extractor = QKRoPEAttentionExtractor(model, [0, 1])
    with extractor.capture():
        outputs = model(ids, attention_mask=mask, position_ids=positions, output_attentions=True)
    for layer in [0, 1]:
        for sample in [0, 1]:
            actual = extractor.compute_attn(layer, torch.tensor([2, 3]), 0, 2,
                                            attention_mask=mask, batch_idx=sample)
            torch.testing.assert_close(actual, outputs.attentions[layer][sample, :, 2:4, :2])


@pytest.mark.parametrize('kwargs,flows', [(None, False), ({'use_reentrant': False}, True)])
def test_trag_gradient_requires_non_reentrant_checkpointing(kwargs, flows):
    """Reentrant checkpointing detaches hooked Q/K, so TRAG would get no gradient."""
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from tinyllava.train.distill_trainer import QKRoPEAttentionExtractor
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=16, hidden_size=16, intermediate_size=32,
                                         num_hidden_layers=2, num_attention_heads=4,
                                         num_key_value_heads=2)).train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=kwargs)  # None -> HF default (reentrant)
    model.enable_input_require_grads()
    extractor = QKRoPEAttentionExtractor(model, [1])
    with extractor.capture():
        model(torch.tensor([[1, 2, 3, 4, 5]]))
    attention = extractor.compute_attn(1, torch.tensor([3, 4]), 0, 2, batch_idx=0)
    assert attention.requires_grad is flows


def test_trainer_forces_non_reentrant_checkpointing_for_trag(tmp_path):
    from transformers import TrainingArguments
    student, teacher = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
    args = TrainingArguments(output_dir=str(tmp_path), gradient_checkpointing=True, no_cuda=True, report_to=[])
    trainer = DistillLLaVATrainer(teacher_model=teacher, distill_mode='trag', model=student, args=args)
    assert trainer.args.gradient_checkpointing_kwargs == {'use_reentrant': False}
