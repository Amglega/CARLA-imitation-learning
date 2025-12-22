import openvino as ov
import torch
from utils.pilotnet import PilotNet
from pathlib import Path
from PIL import Image

import torchvision


MODEL_DIR = Path("openvino_models")
MODEL_DIR.mkdir(exist_ok=True)
MODEL_NAME = "mobilenet"

image_shape = (66, 200, 3)
num_labels = 2
input_size =[66, 200]

model = torchvision.models.mobilenet_v3_large()
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
