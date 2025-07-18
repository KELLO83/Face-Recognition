import torch
import timm
import cv2
import numpy
import torchinfo

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom
import torch.nn as nn



model = timm.create_model('vit_base_patch16_224',pretrained=True)

model = model.to(device='cuda:0')
model.head = nn.Identity() 



# print("=== 기본 summary ===")
# torchinfo.summary(model, input_size=(1, 3, 224, 224))

# print("\n=== 상세 정보 포함 summary ===")
# torchinfo.summary(
#     model, 
#     input_size=(1, 3, 224, 224),
#     col_names=["input_size", "output_size", "num_params", "mult_adds"],
#     depth=3,
#     verbose=1
# )

attn = []
def get_attention_hook(module , input , output):
    print("Call get_hook")
    print("get_attention_hook out shape : " ,output.shape)
    attn.append(output[1])

handle = model.blocks[-1].attn.attn_drop.register_forward_hook(get_attention_hook)


image_path = 'data/cat.jpeg'  
image = cv2.imread(image_path, cv2.IMREAD_COLOR)

print("원본이미지 크기 :",image.shape)

from PIL import Image
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
PIL_image = Image.fromarray(image_rgb)


transform = timm.data.create_transform(**timm.data.resolve_data_config(model.pretrained_cfg))


image_transformed = transform(PIL_image)


input_tensor = image_transformed.unsqueeze(0).to(device='cuda:0') 

model.train()
with torch.no_grad():
    output_all_tokens = model.forward_features(input_tensor)
    cls_token = model(input_tensor)
handle.remove()


print(output_all_tokens.shape)
print(cls_token.shape)



