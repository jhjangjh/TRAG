"""Smoke tests: every distillation objective runs and back-propagates."""
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from tinyllava.train import distill_trainer as module
from tinyllava.train.distill_trainer import DistillLLaVATrainer

B, S, V, H = 2, 7, 5, 2
EXPANDED_LABELS = torch.tensor([[-100, -100, -100, -100, 1, 2, 3],
                                [-100, -100, -100, 4, 0, -100, -100]])


class Model(torch.nn.Module):
    def __init__(self, seed, num_layers, two_outputs=False):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.config = SimpleNamespace(num_hidden_layers=num_layers)
        self.logits = torch.nn.Parameter(torch.randn(B, S, V, generator=g))
        self.raw = torch.nn.Parameter(torch.rand(num_layers, B, H, 3, generator=g) + 0.05)
        self.attn = torch.nn.Parameter(torch.randn(num_layers, B, H, S, S, generator=g))
        self.two_outputs = two_outputs

    def forward(self, output_attentions=False, **kwargs):
        out = {"loss": (self.logits ** 2).mean(), "logits": self.logits}
        if output_attentions:
            out["attentions"] = tuple(a.softmax(-1) for a in self.attn)
        if self.two_outputs:
            return out, EXPANDED_LABELS
        return out, EXPANDED_LABELS, None, None, None, [1, 1], [[(1, 4)], [(1, 4)]]


class Extractor:
    def __init__(self, model, layer_indices):
        self.model = model
        self.registered_layers = list(layer_indices)
        self.q_cache = {l: torch.zeros(B, S, 1) for l in layer_indices}

    def capture(self):
        return nullcontext()

    def compute_attn(self, layer_idx, q_indices, batch_idx, **kwargs):
        row = self.model.raw[layer_idx, batch_idx]
        return row[:, None, :].expand(H, len(q_indices), 3)


def make_trainer(mode):
    trainer = object.__new__(DistillLLaVATrainer)
    trainer.distill_mode = mode
    trainer.entropy_min_ema = trainer.entropy_max_ema = None
    trainer.entropy_ema_momentum = 0.995
    trainer.entropy_warmup_steps = 100
    trainer.entropy_buffer, trainer.entropy_buffer_max = [], 500
    trainer.entropy_fallback_min, trainer.entropy_fallback_max = 2.5, 7.0
    trainer.niter = 0
    trainer.model = Model(1, 24, two_outputs=mode == "compodistill")
    trainer.teacher_model = Model(2, 28).requires_grad_(False)
    trainer.args = SimpleNamespace(local_rank=-1, past_index=-1)
    trainer.label_smoother = None
    trainer.loss_fct = torch.nn.KLDivLoss(reduction="batchmean")
    trainer.train_stage = "dft"
    trainer.use_attn_loss = False
    if mode == "compodistill":
        trainer._init_compodistill_targets()
        trainer.use_attn_loss = True
    return trainer


@pytest.mark.parametrize("mode", ["kd", "llava-kd", "compodistill", "trag"])
def test_objective_runs_and_backpropagates(monkeypatch, mode):
    monkeypatch.setattr(module, "QKRoPEAttentionExtractor", Extractor)
    trainer = make_trainer(mode)
    inputs = {"input_ids": torch.tensor([[-200, 0, 1, 2, 3], [-200, 0, 4, 0, 1]]),
              "labels": EXPANDED_LABELS[:, 2:].clone(), "attention_mask": torch.ones(2, 5)}
    loss = trainer.compute_loss(trainer.model, inputs)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(trainer.model.logits.grad).all()
    if mode == "trag":
        assert trainer.model.raw.grad.abs().sum() > 0
    if mode == "compodistill":
        assert trainer.model.attn.grad.abs().sum() > 0


@pytest.mark.parametrize("student,teacher,group", [(24, 24, 1), (24, 28, 3), (24, 40, 9),
                                                   (28, 28, 1), (28, 36, 5)])
def test_compodistill_teacher_group(student, teacher, group):
    trainer = object.__new__(DistillLLaVATrainer)
    trainer.model = SimpleNamespace(config=SimpleNamespace(num_hidden_layers=student))
    trainer.teacher_model = SimpleNamespace(config=SimpleNamespace(num_hidden_layers=teacher))
    trainer._init_compodistill_targets()
    assert trainer.teacher_group == group
    assert len(trainer.teacher_target_layers) - len(trainer.student_target_layers) + 1 == group


def test_compodistill_rejects_shallower_teacher():
    trainer = object.__new__(DistillLLaVATrainer)
    trainer.model = SimpleNamespace(config=SimpleNamespace(num_hidden_layers=28))
    trainer.teacher_model = SimpleNamespace(config=SimpleNamespace(num_hidden_layers=24))
    with pytest.raises(AssertionError):
        trainer._init_compodistill_targets()


def test_unknown_mode_is_rejected():
    trainer = make_trainer("kd")
    trainer.distill_mode = "unknown"
    with pytest.raises(ValueError):
        trainer.compute_loss(trainer.model, {})


def test_distill_sampler_is_a_permutation_with_pure_full_batches():
    from tinyllava.train.distill_trainer import get_distill_modality_grouped_indices
    batch, world = 2, 1
    # Multiples of the megabatch size: every batch is modality-pure.
    lengths = [5, -3, 7, -2, 4, 6, -8, 9, 3, 2, -1, 8]  # 8 multimodal, 4 text-only
    order = get_distill_modality_grouped_indices(lengths, batch, world)
    assert sorted(order) == list(range(len(lengths)))
    for i in range(0, len(order), batch):
        assert len({lengths[j] > 0 for j in order[i:i + batch]}) == 1
    # Ragged counts: still a permutation; tails are appended at the end.
    ragged = [5, -3, 7, -2, 4, 6, -8, 9, 3, -1, 2]
    order = get_distill_modality_grouped_indices(ragged, batch, world)
    assert sorted(order) == list(range(len(ragged)))


def test_distill_sampler_handles_single_modality():
    from tinyllava.train.distill_trainer import get_distill_modality_grouped_indices
    order = get_distill_modality_grouped_indices([3, 1, 2, 5], 2, 1)
    assert sorted(order) == [0, 1, 2, 3]


def test_trag_queries_follow_truncated_expanded_sequence():
    """Response queries must come from the model's expanded/truncated labels."""
    from tinyllava.model.modeling_tinyllava_distill import TinyLlavaDistillForConditionalGeneration

    visual, dim, max_len = 4, 3, 12
    embed = torch.nn.Embedding(20, dim)
    stub = SimpleNamespace(
        vision_tower=object(), encode_images=lambda im: torch.zeros(im.shape[0], visual, dim),
        language_model=SimpleNamespace(get_input_embeddings=lambda: embed),
        config=SimpleNamespace(tokenizer_model_max_length=max_len, tokenizer_padding_side="right"),
        device=torch.device("cpu"))
    # [IMG, 4 prompt, 5 response] -> 13 expanded tokens, truncated to 12.
    ids = torch.tensor([[-200, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    labels = torch.tensor([[-100] * 5 + [5, 6, 7, 8, 9]])
    out = TinyLlavaDistillForConditionalGeneration.prepare_inputs_labels_for_multimodal(
        stub, ids, None, torch.ones_like(ids), None, labels, torch.zeros(1, 3, 2, 2))
    exp_mask, exp_labels, num_images, spans = out[2], out[5], out[7], out[8]
    assert exp_labels.shape[1] == max_len and spans == [[(0, visual)]]

    seen = {}

    class Recorder:
        registered_layers = [0]
        q_cache = {0: torch.zeros(1, max_len, 1)}

        def compute_attn(self, q_indices, **kwargs):
            seen["q"] = q_indices.tolist()
            return torch.full((1, len(q_indices), visual), 1.0 / visual)

    trainer = make_trainer("trag")
    samples = trainer._trag_attention_samples(exp_labels, exp_mask, Recorder(), Recorder(), [0], [0],
                                              num_images, spans, num_images, spans)
    assert len(samples) == 1
    # Positions 8..11 hold the four response tokens that survive truncation.
    assert seen["q"] == [8, 9, 10, 11]


@pytest.mark.parametrize("pretrained,unified", [(None, False), ("/ckpt/dpt", True)])
def test_recipe_save_format_depends_on_stage_not_path(tmp_path, pretrained, unified):
    from tinyllava.training_recipe.base import BaseTrainingRecipe

    calls = []
    args = SimpleNamespace(output_dir=str(tmp_path / "sft"), pretrained_model_path=pretrained)
    recipe = object.__new__(BaseTrainingRecipe)
    recipe.training_arguments = args
    model = SimpleNamespace(config=SimpleNamespace(save_pretrained=lambda *a, **k: None),
                            tokenizer=SimpleNamespace(save_pretrained=lambda *a, **k: None))
    trainer = SimpleNamespace(deepspeed=None, save_state=lambda: None,
                              save_model=lambda out: calls.append(out))
    if unified:
        recipe.save(model, trainer)
        assert calls == [args.output_dir]
    else:
        with pytest.raises(AttributeError):  # proceeds to the split (component) save
            recipe.save(model, trainer)
        assert calls == []


@pytest.mark.parametrize("sharded", [False, True])
def test_recipe_load_reads_single_and_sharded_unified_checkpoints(tmp_path, sharded):
    import json
    from safetensors.torch import save_file
    from tinyllava.training_recipe.base import BaseTrainingRecipe

    source, target = torch.nn.Linear(3, 2), torch.nn.Linear(3, 2)
    weights = {k: v.detach().clone() for k, v in source.state_dict().items()}
    if sharded:
        save_file({"weight": weights["weight"]}, str(tmp_path / "model-00001-of-00002.safetensors"))
        save_file({"bias": weights["bias"]}, str(tmp_path / "model-00002-of-00002.safetensors"))
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
            "weight": "model-00001-of-00002.safetensors", "bias": "model-00002-of-00002.safetensors"}}))
    else:
        save_file(weights, str(tmp_path / "model.safetensors"))
    recipe = object.__new__(BaseTrainingRecipe)
    recipe.training_arguments = SimpleNamespace(pretrained_model_path=str(tmp_path))
    loaded = recipe.load(target)
    for key, value in weights.items():
        torch.testing.assert_close(loaded.state_dict()[key], value)


def test_sft_sampler_handles_single_modality():
    from tinyllava.train.tinyllava_trainer import get_modality_length_grouped_indices
    for lengths in ([3, 1, 2, 5], [-3, -1, -2, -5]):
        assert sorted(get_modality_length_grouped_indices(lengths, 2, 1)) == [0, 1, 2, 3]


def test_text_only_sample_keeps_connector_in_graph():
    """A text-only micro-batch must still produce (zero) connector gradients."""
    from tinyllava.model.modeling_tinyllava_distill import TinyLlavaDistillForConditionalGeneration

    connector = torch.nn.Linear(3, 3)
    embed = torch.nn.Embedding(20, 3)
    stub = SimpleNamespace(
        vision_tower=object(), encode_images=lambda im: connector(torch.zeros(im.shape[0], 4, 3)),
        language_model=SimpleNamespace(get_input_embeddings=lambda: embed),
        config=SimpleNamespace(tokenizer_model_max_length=16, tokenizer_padding_side="right"),
        device=torch.device("cpu"))
    ids = torch.tensor([[1, 2, 3, 4]])  # no image token
    out = TinyLlavaDistillForConditionalGeneration.prepare_inputs_labels_for_multimodal(
        stub, ids, None, torch.ones_like(ids), None, ids.clone(), torch.zeros(1, 3, 2, 2))
    embeds = out[4]
    assert embeds.shape[1] == 4
    embeds.sum().backward()
    assert connector.weight.grad is not None and torch.count_nonzero(connector.weight.grad) == 0


CAPTIONS = ("A dog on a sofa.", "A cat on a sofa.")


@pytest.mark.parametrize("text,choice", [
    # 1. caption index
    ("(1)", 1), ("(2)", 2), ("1", 1), ("2.", 2), (" (2) A dog on a sofa.", 2), ("Answer: (1)", 1),
    # 2. verbatim caption (case, spacing and punctuation ignored)
    ("A dog on a sofa.", 1), ("a cat on a  sofa", 2),
    # 3. letter
    ("A", 1), ("b.", 2), ("(B)", 2), ("Option A", 1),
    # unparseable -> incorrect
    ("", None), ("   ", None), ("(1) or (2)", None), ("A dog.", None), ("A cat on a sofa near a dog.", None),
    ("Both sentences are correct.", None), ("12 dogs", None), ("AB", None),
])
def test_cr_parser_accepts_only_a_clear_selection(text, choice):
    from tinyllava.eval.choice_parsing import parse_choice
    assert parse_choice(text, *CAPTIONS) == choice


def test_cr_parser_rejects_identical_captions_and_missing_options():
    from tinyllava.eval.choice_parsing import parse_choice
    assert parse_choice("a dog", "A dog", "a dog.") is None
    assert parse_choice("A dog on a sofa.") is None  # no captions given: only index/letter rules


def test_cr_empty_responses_score_zero():
    from tinyllava.eval.choice_parsing import parse_choice
    from tinyllava.eval.model_vqa_sugarcrepe import calculate_accuracy
    preds = [{"split": "add_att", "variant": v, "ground_truth": g, "predicted": parse_choice("", *CAPTIONS),
              "correct": parse_choice("", *CAPTIONS) == g} for v, g in (("positive_first", 1), ("negative_first", 2))]
    accuracy = calculate_accuracy(preds)
    assert all(not p["correct"] for p in preds)
    assert accuracy["overall_average"]["accuracy"] == 0.0


@pytest.mark.parametrize("options,expected", [
    ({}, {}),  # full precision on CPU (the reported evaluation setting)
    ({"torch_dtype": torch.float16}, {"torch_dtype": torch.float16}),
    ({"load_8bit": True, "device": "cuda:1"}, {"torch_dtype": torch.float16, "device_map": {"": "cuda:1"}}),
])
def test_eval_loader_passes_precision_and_quantization(monkeypatch, tmp_path, options, expected):
    from tinyllava.model import load_model
    seen = {}

    def fake_from_pretrained(path, **kwargs):
        seen.update(kwargs)
        raise StopIteration  # stop after recording the arguments

    monkeypatch.setattr(load_model.TinyLlavaForConditionalGeneration, "from_pretrained", fake_from_pretrained)
    with pytest.raises(StopIteration):
        load_model.load_pretrained_model(str(tmp_path), **options)
    for key, value in expected.items():
        assert seen[key] == value
    assert ("quantization_config" in seen) == bool(options.get("load_8bit"))
    if not options:
        assert "torch_dtype" not in seen and "device_map" not in seen
