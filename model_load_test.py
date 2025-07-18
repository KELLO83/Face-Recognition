import torch
import models.backbone
import torchinfo
# 가중치 파일 로드
weight = torch.load('models/weight/ms1mv3_arcface_r50_fp16.pth', map_location='cpu')


iresnet_model = models.backbone.iresnet50()
print(iresnet_model)
load_result = iresnet_model.load_state_dict(weight, strict=False)


print("누락된 가중치 : {}".format(load_result.missing_keys))
print("예상치못한 가중치 : {}".format(load_result.unexpected_keys))

if not load_result.missing_keys and not load_result.unexpected_keys:
    print("모델 가중치가 성공적으로 로드되었습니다.")


