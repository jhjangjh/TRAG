# Modified by the TRAG authors from TinyLLaVA Factory. See NOTICE.
import os

import torch
import torch.nn as nn


class Connector(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self._connector = None

    def load_model(self, **kwargs):
        pretrained_connector_path = kwargs.get('pretrained_connector_path', None)
        if pretrained_connector_path is not None:
            # Try safetensors first, then fall back to pytorch_model.bin
            safetensor_path = os.path.join(pretrained_connector_path, 'model.safetensors')
            bin_path = os.path.join(pretrained_connector_path, 'pytorch_model.bin')

            if os.path.exists(safetensor_path):
                # Load from safetensors
                from safetensors.torch import load_file
                connector_weights = load_file(safetensor_path)
                print(f'Loading connector from {safetensor_path}...')
            elif os.path.exists(bin_path):
                # Load from pytorch_model.bin
                connector_weights = torch.load(bin_path, map_location='cpu')
                print(f'Loading connector from {bin_path}...')
            else:
                raise FileNotFoundError(
                    f"No connector weights found at {pretrained_connector_path}. "
                    f"Tried: model.safetensors, pytorch_model.bin"
                )

            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
            self._connector.load_state_dict(get_w(connector_weights, '_connector'))

        for p in self._connector.parameters():
            p.requires_grad = False




    def forward(self, x):
        return self._connector(x)
