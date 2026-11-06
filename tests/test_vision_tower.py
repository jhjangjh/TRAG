"""SigLIP keeps every patch token; CLIP-style towers drop the CLS token."""
import torch
from transformers import SiglipVisionConfig, SiglipVisionModel

from tinyllava.model.vision_tower.base import VisionTower
from tinyllava.model.vision_tower.siglip import SIGLIPVisionTower


def tiny_siglip():
    config = SiglipVisionConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                                num_attention_heads=2, image_size=27, patch_size=3)
    return SiglipVisionModel(config).eval()


def test_siglip_keeps_all_patch_tokens():
    tower = torch.nn.Module.__new__(SIGLIPVisionTower)
    torch.nn.Module.__init__(tower)
    tower._vision_tower = tiny_siglip()
    features = tower(torch.zeros(1, 3, 27, 27), vision_feature_select_strategy="patch")
    assert features.shape[1] == (27 // 3) ** 2  # 81; 729 for SigLIP-SO400M/14 at 384px


def test_base_tower_drops_cls_token():
    tower = torch.nn.Module.__new__(VisionTower)
    torch.nn.Module.__init__(tower)
    tower._vision_tower = tiny_siglip()
    features = tower(torch.zeros(1, 3, 27, 27), vision_feature_select_strategy="patch")
    assert features.shape[1] == (27 // 3) ** 2 - 1
