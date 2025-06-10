# dynamic_erasing.py

import torch
import torch.nn as nn

class DynamicErasing(nn.Module):
    def __init__(self, erase_prob=0.5, erase_ratio=0.2):
        super(DynamicErasing, self).__init__()
        self.erase_prob = erase_prob
        self.erase_ratio = erase_ratio

    def forward(self, x):
        if not self.training or torch.rand(1).item() > self.erase_prob:
            return x

        B, C, H, W = x.size()
        act_map = x.mean(1, keepdim=True)
        flat = act_map.view(B, -1)
        topk = int(H * W * self.erase_ratio)

        mask = torch.ones_like(act_map)
        for i in range(B):
            _, idxs = flat[i].topk(topk)
            y = idxs // W
            x_coord = idxs % W
            mask[i, 0, y, x_coord] = 0

        x = x * mask
        return x
