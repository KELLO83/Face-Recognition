
import torch.nn as nn
import math
from .focal_loss import FocalLoss
import torch

class ArcFaceLoss(nn.modules.Module):
    def __init__(self, s=45.0, m=0.1, crit="bce"):
        super().__init__()
        if crit == "focal":
            self.crit = FocalLoss(gamma=2.0)
        elif crit == "bce":
            self.crit = nn.CrossEntropyLoss(reduction="none")   
        self.s = s

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        
    def forward(self, cosine, labels):
        logits = logits.float()
        cosine = logits
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        labels2 = torch.zeros_like(cosine)
        labels2.scatter_(1, labels.view(-1, 1).long(), 1)
        output = (labels2 * phi) + ((1.0 - labels2) * cosine)

        s = self.s
        output = output * s
        loss = self.crit(output, labels)
        return loss.mean()