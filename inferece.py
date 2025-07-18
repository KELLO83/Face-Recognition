import torch
from models.backbone.irsnet import iresnet50





backbone = iresnet50()
head = 
backbone_pretrained_path = ''
head_pretrained_path = ''

bacbkbone_model = torch.load(backbone_pretrained_path , map_location = 'cpu')
