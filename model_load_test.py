import torch
from models.backbone.ir_ASIS_Resnet import IR_50 , IR_101
import torchinfo
# 가중치 파일 로드
weight = torch.load('models/weight/backbone_ir50_asia.pth', map_location='cpu')


iresnet_model = IR_101(input_size=(112, 112, 3))
print(iresnet_model)
load_result = iresnet_model.load_state_dict(weight, strict=False)


print("누락된 가중치 : {}".format(load_result.missing_keys))
print("예상치못한 가중치 : {}".format(load_result.unexpected_keys))

if not load_result.missing_keys and not load_result.unexpected_keys:
    print("모델 가중치가 성공적으로 로드되었습니다.")


    