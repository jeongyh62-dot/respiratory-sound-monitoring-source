from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    """Grad-CAM helper compatible with AlexNet in-place ReLU behavior."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        for module in self.model.modules():
            if isinstance(module, nn.ReLU):
                module.inplace = False
        self.forward_handle = target_layer.register_forward_hook(self._forward_hook)

    def _save_gradient(self, grad):
        self.gradients = grad.detach()

    def _forward_hook(self, module, inputs, output):
        self.activations = output.detach()
        if output.requires_grad:
            output.register_hook(self._save_gradient)

    def generate(self, input_tensor: torch.Tensor, target_class: int | None = None):
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        output = self.model(input_tensor)
        if target_class is None:
            target_class = int(output.argmax(dim=1).item())
        output[0, target_class].backward()
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].detach().cpu()
        cmin, cmax = cam.min(), cam.max()
        if cmax > cmin:
            cam = (cam - cmin) / (cmax - cmin)
        return cam, output.detach()

    def close(self):
        self.forward_handle.remove()
