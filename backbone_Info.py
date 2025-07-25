from models.metrics import ArcMarginProduct
import torch
import torchinfo
from models.backbone.ir_ASIS_Resnet  import Backbone





model = Backbone(input_size=(112, 112, 3), num_layers=50, mode='ir_se')

torchinfo.summary(model, input_size=(1, 3, 112, 112), col_names=["input_size", "output_size", "num_params", "trainable"], row_settings=["var_names"])
