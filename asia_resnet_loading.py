
import torch
from models.backbone.ir_ASIS_Resnet import Backbone
import torchinfo

weight2 = torch.load('models/weight/backbone_ir50_asia.pth', map_location='cpu')
weight = torch.load('models/weight/adaface_ir50_webface4m.ckpt' , map_location='cpu')
weight = weight['state_dict']
backbone = Backbone(input_size=(112, 112, 3), num_layers=50)
load_result = backbone.load_state_dict(weight, strict=False)


print("누락된 가중치 : {}".format(load_result.missing_keys))
print("예상치못한 가중치 : {}".format(load_result.unexpected_keys))

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


backbone_params = sum(p.numel() for p in backbone.parameters()) 
trainable_backbone_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
trainable_percentage = (trainable_backbone_params / backbone_params) * 100 if backbone_params > 0 else 0


print(f"Total Backbone Parameters: {backbone_params}")
print(f"Trainable Backbone Parameters: {trainable_backbone_params} ({trainable_percentage:.2f}%)")
print(f"Frozen Backbone Parameters: {backbone_params - trainable_backbone_params} ({100 - trainable_percentage:.2f}%)")

backbone = backbone.to('cuda:1')
backbone.eval()
dummy_input = dummy_input.to('cuda:1')
dummy_input = torch.tensor(dummy_input , dtype=torch.float32)
out = backbone(dummy_input)

print(out.shape)