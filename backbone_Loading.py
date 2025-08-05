import torch
from models.backbone.ir_ASIS_Resnet import Backbone
import torchinfo

weight = torch.load('models/weight/backbone_ir50_asia.pth', map_location='cpu')
# weight = torch.load('models/weight/adaface_ir50_webface4m.ckpt' , map_location='cpu')



if 'backbone_state_dict' in weight:
    state_dict = weight['backbone_state_dict']
elif 'metric_fc_state_dict' in weight:
    state_dict = weight['metric_fc_state_dict']
elif 'state_dict' in weight:
    state_dict = weight['state_dict']
else:
    state_dict = weight


iresnet_model = Backbone(input_size=(112, 112, 3), num_layers=50)
load_result = iresnet_model.load_state_dict(state_dict, strict=False)


print("누락된 가중치 : {}".format(load_result.missing_keys))
print("예상치못한 가중치 : {}".format(load_result.unexpected_keys))

if not load_result.missing_keys and not load_result.unexpected_keys:
    print("모델 가중치가 성공적으로 로드되었습니다.")

for param in iresnet_model.parameters():
    param.requires_grad = True

for name, param in iresnet_model.named_parameters():
    if 'body.' in name:
        try:
            body_idx = int(name.split('.')[1])
            if body_idx < 13:
                param.requires_grad = False
        except (ValueError, IndexError):
            continue
    elif 'input_layer' in name:
        param.requires_grad = False

print("모델 파라미터 동결 상태 확인:")
for name, param in iresnet_model.named_parameters():
    if not param.requires_grad:
        print(f"[Frozen]  {name}")
    else:
        print(f"[Trainable] {name}")

# print("모델의 파라미터 중 학습 가능한 파라미터:")
# for name, param in iresnet_model.named_parameters():
#     if param.requires_grad:
#         print(name, param.shape)



dummy_input = torch.randn(1, 3, 112, 112)
model_info = torchinfo.summary(
    iresnet_model,
    input_size=(1, 3, 112, 112),
    verbose=True,
    col_names=["input_size", "output_size", "num_params", "trainable"],
    row_settings=["var_names"],
    depth=4
)
print(model_info)

# print("모델의 파라미터 중 학습 불가능한 파라미터:")
# for name, param in iresnet_model.named_parameters():
#     if not param.requires_grad:
#         print(name, param.shape)
