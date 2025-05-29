import os
from utils.processing import *
from utils.pilot_net_dataset import PilotNetDataset
from utils.pilotnet import PilotNet
from utils.transform_helper import createTransform
import time
import argparse
from PIL import Image
import cv2
import json
import numpy as np
from copy import deepcopy
from tqdm import tqdm
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import  models,transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", action='append', help="Directory to find Train Data")
    parser.add_argument("--test_dir", action='append', help="Directory to find Test Data")
    parser.add_argument("--preprocess", action='append', default=None, help="preprocessing information: choose from crop/nocrop and normal/extreme")
    parser.add_argument("--base_dir", type=str, default='exp_random', help="Directory to save everything")
    parser.add_argument("--comment", type=str, default='Random Experiment', help="Comment to know the experiment")
    parser.add_argument("--data_augs", action='append', type=str, default=None, help="Data Augmentations")
    parser.add_argument("--mirrored_imgs", type=bool, default=False, help="Add mirrored images to the train data")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of Epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for Policy Net")
    parser.add_argument("--test_split", type=float, default=0.2, help="Train test Split")
    parser.add_argument("--shuffle", type=bool, default=False, help="Shuffle dataset")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--save_iter", type=int, default=50, help="Iterations to save the model")
    parser.add_argument("--print_terminal", type=bool, default=False, help="Print progress in terminal")
    parser.add_argument("--seed", type=int, default=123, help="Seed for reproducing")
    parser.add_argument("--model", type=str, default='pilotnet', help="Model to train")
    parser.add_argument("--pretrained", type=bool, default=False, help="Specify if the model pretrained weights are loaded")

    args = parser.parse_args()
    return args


def my_loss(output, target):
    loss = torch.mean((output - target)**2)
    return loss


if __name__=="__main__":

    args = parse_args()

    exp_setup = vars(args)

    # Base Directory
    path_to_data = args.data_dir
    base_dir = './experiments/'+ args.base_dir + '/'
    model_save_dir = base_dir + 'trained_models'
    log_dir = base_dir + 'log'
    check_path(base_dir)
    check_path(log_dir)
    check_path(model_save_dir)

    print("Saving model in:" + model_save_dir)

    with open(base_dir+'args.json', 'w') as fp:
        json.dump(exp_setup, fp)

    # Hyperparameters
    augmentations = args.data_augs
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    learning_rate = args.lr
    val_split = args.test_split
    shuffle_dataset = args.shuffle
    save_iter = args.save_iter
    random_seed = args.seed
    print_terminal = args.print_terminal
    mirrored_img = args.mirrored_imgs
    # Device Selection (CPU/GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    FLOAT = torch.FloatTensor

    # Tensorboard Initialization
    writer = SummaryWriter(log_dir)

    # Define data transformations
    transformations = createTransform(augmentations)
    # Load data
    dataset = PilotNetDataset(path_to_data, mirrored_img, transformations, preprocessing=args.preprocess)


    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(val_split * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_split = indices[split:], indices[:split]

    train_loader = DataLoader(dataset, batch_size=batch_size)
    val_loader = DataLoader(dataset, batch_size=batch_size)

    # Load Model    
    model_name = args.model
    is_pretrained = args.pretrained
    model_weights = None

    if model_name == 'pilotnet':
        model = PilotNet(dataset.image_shape, dataset.num_labels).to(device)
    elif model_name == 'mobilenet_large':
        if is_pretrained:
            model_weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 
        model = models.mobilenet_v3_large(weights=model_weights)
        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_ftrs, 3)
    elif model_name == 'mobilenet_small':
        if is_pretrained:
            model_weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        model = models.mobilenet_v3_small(weights=model_weights)
        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_ftrs, 3)
    elif model_name == 'resnet':
        if is_pretrained:
            model_weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=model_weights)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 3)

    model.to(device)
    if os.path.isfile( model_save_dir + '/' + model_name + '_model_{}.pth'.format(random_seed)):
        model.load_state_dict(torch.load(model_save_dir + '/' + model_name + '_model_{}.pth'.format(random_seed),map_location=device))
        best_model = deepcopy(model)

    # CSV loss log
    self_path = os.getcwd()
    writer_output = csv.writer(open(self_path + '/train_data_' + model_name + '_{}'.format(random_seed) + '.csv', "w"))
    writer_output.writerow(["epoch", "loss"])
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    total_step = len(train_loader)
    global_iter = 0
    global_val_mse = 0.05


    print("*********** Training Started ************")

    # Store loss values for plotting
    epoch_losses = []

    for epoch in range(0, num_epochs):
        model.train()
        train_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            
            images = FLOAT(images).to(device)

            labels = FLOAT(labels.float()).to(device)

            # Zero the optimizer gradients
            optimizer.zero_grad()
            # Run the forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            current_loss = loss.item()
            train_loss += current_loss
            # Backprop and perform Adam optimisation
            loss.backward()
            optimizer.step()

            if global_iter % save_iter == 0:
                torch.save(model.state_dict(), model_save_dir + '/' + model_name + '_model_{}.pth'.format(random_seed))
            global_iter += 1

            if print_terminal and (i + 1) % 10 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                    .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
        
        # Add entry for last epoch                            
        with open(model_save_dir+'/args.json', 'w') as fp:
            json.dump({'last_epoch': epoch}, fp)
        writer.add_scalar("performance/train_loss", train_loss/len(train_loader), epoch+1)

        # Save epoch loss
        epoch_losses.append(train_loss/len(train_loader))
        
        # Validation 
        model.eval()
        with torch.no_grad():
            val_loss = 0 
            for images, labels in val_loader:
                images = FLOAT(images).to(device)
                labels = FLOAT(labels.float()).to(device)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
                
            val_loss /= len(val_loader) # take average
            writer.add_scalar("performance/valid_loss", val_loss, epoch+1)
        
            writer_output.writerow([epoch+1,val_loss])

        # compare
        if val_loss < global_val_mse:
            global_val_mse = val_loss
            best_model = deepcopy(model)
            torch.save(best_model.state_dict(), model_save_dir + '/' + model_name + '_model_best_{}.pth'.format(random_seed))
            mssg = "Model Improved!!"
        else:
            mssg = "Not Improved!!"

        print('Epoch [{}/{}], Validation Loss: {:.4f}'.format(epoch + 1, num_epochs, val_loss), mssg)
        

    model = best_model # allot the best model on validation 
    # Test the model
    transformations_val = createTransform([]) 
    
    test_set = PilotNetDataset(args.test_dir, mirrored_img, transformations_val, preprocessing=args.preprocess)
    test_loader = DataLoader(test_set, batch_size=batch_size)
    print("Check performance on testset")
    model.eval()
    with torch.no_grad():
        test_loss = 0
        for images, labels in tqdm(test_loader):
            images = FLOAT(images).to(device)
            labels = FLOAT(labels.float()).to(device)
            outputs = model(images)
            test_loss += criterion(outputs, labels).item()
    
    writer.add_scalar('performance/Test_MSE', test_loss/len(test_loader))
    print(f'Test loss: {test_loss/len(test_loader)}')
        
    # Plot loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), epoch_losses, marker='o')
    plt.title('Training Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()