import requests
from pathlib import Path
from PIL import Image
import torch
import torchvision
import numpy as np

MODEL_DIR = Path("openvino_models")
DATA_DIR = Path("data")

MODEL_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
MODEL_NAME = "regnet_y_800mf"

image = Image.open(requests.get("https://farm9.staticflickr.com/8225/8511402100_fea15da1c5_z.jpg", stream=True).raw)

labels_file = DATA_DIR / "imagenet_2012.txt"

if not labels_file.exists():
    resp = requests.get("https://raw.githubusercontent.com/openvinotoolkit/open_model_zoo/master/data/dataset_classes/imagenet_2012.txt")
    with labels_file.open("wb") as f:
        f.write(resp.content)

imagenet_classes = labels_file.open("r").read().splitlines()



# get default weights using available weights Enum for model
weights = torchvision.models.RegNet_Y_800MF_Weights.DEFAULT

# create model topology and load weights
model = torchvision.models.regnet_y_800mf(weights=weights)

# switch model to inference mode
model.eval()


# Initialize the Weight Transforms
preprocess = weights.transforms()

# Apply it to the input image
img_transformed = preprocess(image)

# Add batch dimension to image tensor
input_tensor = img_transformed.unsqueeze(0)

# Perform model inference on input tensor
#result = model(input_tensor)


# Postprocess results
#top_labels, top_scores = postprocess_result(result.detach().numpy())

# Show results
#########print(result)


import openvino as ov

# Create OpenVINO Core object instance
core = ov.Core()

# Convert model to openvino.runtime.Model object
ov_model = ov.convert_model(model)

# Save openvino.runtime.Model object on disk
ov.save_model(ov_model, MODEL_DIR / f"{MODEL_NAME}_dynamic.xml")
print(ov_model)
# Load OpenVINO model on device
compiled_model = core.compile_model(ov_model)
#print(compiled_model)
result = compiled_model(input_tensor)[0]
#print(result)


