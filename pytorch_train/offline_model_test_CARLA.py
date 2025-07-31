#!/usr/bin/env python
import os
import numpy as np
import matplotlib.pyplot as plt
import csv
import argparse
import time
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
import timm
from utils.pilotnet import PilotNet
from datetime import datetime



def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):

    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--test_dir", type=str, help="Directory to find Test Data")
    parser.add_argument("--model_name", type=str, default='pilotnet', help="Model name")
    parser.add_argument("--saved_model_path", type=str, help="Path to the saved model")

    args = parser.parse_args()
    return args

def main():

    # Device Selection (CPU/GPU)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_type = "cuda"
    else:
        device = torch.device("cpu")
        device_type = "cpu"

    data_v_array= []
    net_v_array= []
    data_w_array= []
    net_w_array= []
    #data_b_array = []
    #net_b_array = []
    n_array = []
    count = 1

    args = parse_args()
    path = os.getcwd()
    image_shape = (66, 200, 3)
    num_labels = 2
    input_size =[66, 200]

    # Load the state dictionary from the local .pth file
    state_dict = torch.load(args.saved_model_path,weights_only=True)

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
    elif model_name == 'efficientvit':
        model = timm.create_model('efficientvit_b0', pretrained=False)
        num_ftrs = model.head.classifier[-1].in_features
        model.head.classifier[-1] = nn.Linear(num_ftrs, 2)
    elif model_name == 'fastvit':
        model = timm.create_model('fastvit_mci0', pretrained=False)
        num_ftrs = model.head.classifier[-1].in_features
        model.head.classifier[-1] = nn.Linear(num_ftrs, 2)
    
    model.to(device)

    # Load the state dictionary into the model
    model.load_state_dict(state_dict)
    
    # Set the model to evaluation mode
    model.eval()

    preprocess = transforms.Compose([
        # convert the frame to a CHW torch tensor
        transforms.ToTensor()
    ]) 
    
    num_lines = 0
    with open(path + "/" + args.test_dir + "/data.csv", "rb") as f:
        num_lines = sum(1 for _ in f)

    data_file = open(path + "/" + args.test_dir + "/data.csv", "r")
    reader_csv = csv.reader(data_file) 

    first_line = True
    total_time = 0
    min_time = 20000
    max_time = -1

    printProgressBar(0, num_lines, prefix = 'Progress:', suffix = 'Complete', length = 50)

    total_loss_v = 0
    total_loss_w = 0
    #total_loss_b = 0
    criterion = nn.MSELoss()
    
    start_measure_iter=20

    for line in reader_csv:
        
        if first_line:
            first_line = False
            continue
                    
            
        image = cv2.imread(path + "/" + args.test_dir + "/" + line[0])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #print(path + "/" + args.test_dir + "/" + line[0])
        start_time = datetime.now()

        cropped_image = image[240:480, 0:640]

        resized_image = cv2.resize(cropped_image, (int(input_size[1]), int(input_size[0])))

        # Display cropped image
        #cv2.imshow("image", resized_image)       
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        input_tensor = preprocess(resized_image).to(device)

        #print(type(input_tensor))
        # The model can handle multiple images simultaneously so we need to add an
        # empty dimension for the batch.
        # [3, 200, 66] -> [1, 3, 200, 66]
        input_batch = input_tensor.unsqueeze(0).to(device)
        # Inference (min 20hz max 200hz)

        output = model(input_batch)

        if device_type == "cpu":
            net_v_array.append(output[0].detach().numpy()[0])
            net_w_array.append(output[0].detach().numpy()[1])
            #net_b_array.append(output[0].detach().numpy()[2])

            total_loss_v = total_loss_v + abs(float(line[1])-output[0].detach().numpy()[0]) 
            total_loss_w = total_loss_w + abs(float(line[2])-output[0].detach().numpy()[1])
            #total_loss_b = total_loss_b + abs(float(line[3])-output[0].detach().numpy()[2])
        else:
            net_v_array.append(output.data.cpu().numpy()[0][0])
            net_w_array.append(output.data.cpu().numpy()[0][1])
            #net_b_array.append(output.data.cpu().numpy()[0][2])

            total_loss_v = total_loss_v +  abs(float(line[1])-output.data.cpu().numpy()[0][0]) 
            total_loss_w = total_loss_w +  abs(float(line[2])-output.data.cpu().numpy()[0][1])
            #total_loss_b = total_loss_b +  abs(float(line[3])-output.data.cpu().numpy()[0][2])


        data_v_array.append(float(line[1]))
        data_w_array.append(float(line[2]))
        #data_b_array.append(float(line[3]))
        n_array.append(count)
        
        finish_time = datetime.now()
        dt = finish_time - start_time
        ms = (dt.days * 24 * 60 * 60 + dt.seconds) * 1000 + dt.microseconds / 1000.0
        if count > start_measure_iter:
            total_time = total_time + ms
            if ms < min_time:
                min_time = ms
            if ms > max_time:
                max_time = ms
      
        count = count + 1
        printProgressBar(count, num_lines, prefix = 'Progress:', suffix = 'Complete', length = 50)

    
    
    data_file.close()

    print("Tiempo medio:"+str(total_time/count))
    print("Error medio throttle:"+str(total_loss_v/count))
    print("Error medio steer:"+str(total_loss_w/count))
    #print("Error medio brake:"+str(total_loss_b/count))
    print("Tiempo min:"+str(min_time))
    print("Tiempo max:"+str(max_time))


    
    plt.subplot(1, 2, 1)
    plt.plot(n_array, data_v_array, label = "controller", color='b')
    plt.plot(n_array, net_v_array, label = "net", color='tab:orange')
    plt.title("Linear speed comparison") 
    plt.xlabel('Samples')
    plt.ylabel('Linear speed output')
    plt.legend(loc="upper left")

    plt.subplot(1, 2, 2)
    plt.plot(n_array, data_w_array, label = "controller", color='b')
    plt.plot(n_array, net_w_array, label = "net", color='tab:orange')
    plt.title("Angular speed comparison")
    plt.xlabel('Samples')
    plt.ylabel('Angular speed output') 
    plt.legend(loc="upper left")

    #plt.subplot(1, 3, 3)
    #plt.plot(n_array, data_b_array, label = "controller", color='b')
    #plt.plot(n_array, net_b_array, label = "net", color='tab:orange')
    #plt.title("Brake comparison")
    #plt.xlabel('Samples')
    #plt.ylabel('Brake output') 
    #plt.legend(loc="upper left")
    
    plt.show()

    print("FIN")
    



# Execute!
if __name__ == "__main__":
    main()
