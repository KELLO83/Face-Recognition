
import torch
from models.backbone.ir_ASIS_Resnet import Backbone
import torchinfo

weight = torch.load('models/weight/backbone_ir50_asia.pth', map_location='cpu')


backbone = Backbone(input_size=(112, 112, 3), num_layers=50)
print(backbone)
load_result = backbone.load_state_dict(weight, strict=False)


for name, param in backbone.named_parameters():
    if 'body.' in name:
        try:
            body_idx = int(name.split('.')[1])
            if body_idx < 20: 
                param.requires_grad = False
        except (ValueError, IndexError):
            continue
    elif 'input_layer' in name:
        param.requires_grad = False

print("모델 파라미터 동결 상태 확인:")
for name, param in backbone.named_parameters():
    if not param.requires_grad:
        print(f"[Frozen]  {name}")
    else:
        print(f"[Trainable] {name}")


dummy_input = torch.randn(1, 3, 112, 112)
model_info = torchinfo.summary(
    backbone,
    input_size=(1, 3, 112, 112),
    verbose=True,
    col_names=["input_size", "output_size", "num_params", "trainable"],
    row_settings=["var_names"],
    depth=4
)
print(model_info)