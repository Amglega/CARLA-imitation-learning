#!/usr/bin/env python
import os
import numpy as np
import cv2
import argparse
import time
import utils.hal as HAL
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from utils.pilotnet import PilotNet

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--net_name", type=str, default=None, help="Path to saved model")
    parser.add_argument("--model_name", type=str, default='pilotnet', help="Model name")
    parser.add_argument("--saved_model_path", type=str, help="Path to the saved model")
    parser.add_argument("--OpenVINO", type=bool, default=False, help="Use OpenVINO or not")

    args = parser.parse_args()
    return args

args = parse_args()

image_shape = (66, 200, 3)
num_labels = 2
input_size =[66, 200]

if torch.cuda.is_available():
    device = torch.device("cuda")
    device_type = "cuda"
else:
    device = torch.device("cpu")
    device_type = "cpu"

# Load the state dictionary from the local .pth file
state_dict = torch.load(args.saved_model_path, map_location = device)

model_name = args.model_name

if model_name == 'pilotnet':
    model = PilotNet(image_shape, num_labels)
elif model_name == 'mobilenet_large':
    model = models.mobilenet_v3_large()
    num_ftrs = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(num_ftrs, 2)
elif model_name == 'mobilenet_small':
    model = models.mobilenet_v3_small()
    num_ftrs = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(num_ftrs, 2)
elif model_name == 'resnet':
    model = models.resnet18()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
elif model_name == 'efficientnet_v2':
    model = models.efficientnet_v2_s(weights=None)
    num_ftrs = model.classifier[-1].in_features
    model.classifier[-1] = torch.nn.Linear(num_ftrs, 2)

model.to(device)

# Load the state dictionary into the model
model.load_state_dict(state_dict)
    
# Set the model to evaluation mode
model.eval()

preprocess = transforms.Compose([
    # convert the frame to a CHW torch tensor
    transforms.ToTensor()
])


if args.OpenVINO:
    # Create OpenVINO Core object instance
    core = ov.Core()

    # Convert model to openvino.runtime.Model object
    ov_model = ov.convert_model(model)

    # Save openvino.runtime.Model object on disk
    #ov.save_model(ov_model, "./" + model_name + "_dynamic.xml")

    # Load OpenVINO model on device
    compiled_model = core.compile_model(ov_model)
        
    model = None

def user_main():
    image= HAL.getImage()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # height, width
    height = image.shape[0]


    #crop image
    if height > 100:
        
        cropped_image = image[240:480, 0:640]

        resized_image = cv2.resize(cropped_image, (int(input_size[1]), int(input_size[0])))

        """
        # Display cropped image

        #cv2.imshow("cropped", resized_image)
        #cv2.waitKey(1)
        input_tensor = resized_image.reshape((1, 3, input_size[0], input_size[1])).astype(np.float32)
        # Inference (min 20hz max 200hz)
        ort_inputs = {ort_session.get_inputs()[0].name: input_tensor}
        output = ort_session.run(None, ort_inputs)[0][0]
        print(output)
        """

        input_tensor = preprocess(resized_image).to(device)
        input_batch = input_tensor.unsqueeze(0)
        if args.OpenVINO:
            output = compiled_model(input_batch)[0]
            v = output[0][0]
            w = output[0][1]
        else:
            output = model(input_batch)
            v = output[0].detach().numpy()[0]
            w = output[0].detach().numpy()[1]

        HAL.setV(v)
        HAL.setW(w)

def main():

    HAL.setW(0)
    HAL.setV(0)
    HAL.main(user_main)

# Execute!
if __name__ == "__main__":
    main()
