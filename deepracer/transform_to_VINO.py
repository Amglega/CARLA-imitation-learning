import openvino as ov
import torch
from utils.pilotnet import PilotNet
from pathlib import Path
from PIL import Image
import timm
import torchvision


MODEL_DIR = Path("my_vit_model")
MODEL_DIR.mkdir(exist_ok=True)
MODEL_NAME = "my_vit_model"

image_shape = (66, 200, 3)
num_labels = 2
input_size =[66, 200]

model = timm.create_model("fastvit_sa12")
num_ftrs=model.head.fc.in_features
model.head.fc= torch.nn.Linear(num_ftrs,2)
model.eval()


# Create OpenVINO Core object instance
core = ov.Core()

# Convert model to openvino.runtime.Model object
ov_model = ov.convert_model(model)

# Save openvino.runtime.Model object on disk
ov.save_model(ov_model, MODEL_DIR / f"{MODEL_NAME}_dynamic.xml")

print(ov_model)

compiled_model = core.compile_model(ov_model,'CPU')

print(compiled_model)
