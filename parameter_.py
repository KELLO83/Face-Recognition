from models.backbone.irsnet import iresnet100
import torch
model = iresnet100()


for name , param in model.named_parameters():
    print(f"{name} : {param.shape}")



for child in model.children():
    print("======")
    print(child)


optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)


for name in optimizer.state_dict():
    print(f"{name} ",optimizer.state_dict()[name])