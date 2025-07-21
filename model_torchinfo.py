from models.metrics import ArcMarginProduct
import torch
from torchinfo import summary

# 모델 생성
Head = ArcMarginProduct(512, 1000, m=0.5).to('cuda')

summary(
    Head,
    input_size=[(2, 512), (2,)], 
    dtypes=[torch.float32, torch.long],  
    col_names=[
        "input_size", 
        "output_size", 
        "num_params", 
        "params_percent",
        "kernel_size",
        "mult_adds",
        "trainable"
    ],
    verbose=2, 
    depth=5, 
    device='cuda'
)
