import os
import csv
import torch
import torch.nn as nn
from torchvision import  models,transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
import matplotlib.pyplot as plt
from utils.processing import *
from utils.pilot_net_dataset import PilotNetDataset
from utils.pilotnet import PilotNet
from utils.transform_helper import createTransform
import argparse
import json
import numpy as np
from copy import deepcopy
from tqdm import tqdm
import timm
from sklearn.model_selection import KFold

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_dir", action='append', help="Directory to find Train Data")
    parser.add_argument("--num_folds", type=int, default=5, help="Number of folds for cross validation")
    parser.add_argument("--test_dir", action='append', help="Directory to find Test Data")
    parser.add_argument("--preprocess", action='append', default=None, help="preprocessing information: choose from crop/nocrop and normal/extreme")
    parser.add_argument("--base_dir", type=str, default='exp_random', help="Directory to save everything")
    parser.add_argument("--comment", type=str, default='Random Experiment', help="Comment to know the experiment")
    parser.add_argument("--data_augs", action='append', type=str, default=None, help="Data Augmentations")
    parser.add_argument("--mirrored_imgs", type=bool, default=False, help="Add mirrored images to the train data")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of Epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for Policy Net")
    parser.add_argument("--test_split", type=float, default=0.2, help="Train test Split")
    parser.add_argument("--shuffle", type=bool, default=True, help="Shuffle dataset")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--save_iter", type=int, default=50, help="Iterations to save the model")
    parser.add_argument("--print_terminal", type=bool, default=False, help="Print progress in terminal")
    parser.add_argument("--seed", type=int, default=46, help="Seed for reproducing results")
    parser.add_argument("--model", type=str, default='pilotnet', help="Model to train")
    parser.add_argument("--pretrained", type=bool, default=False, help="Specify if the model pretrained weights are loaded")

    args = parser.parse_args()
    return args

if __name__=="__main__":
	
    args = parse_args()

    exp_setup = vars(args)

    # Base Directory
    path_to_data = args.train_dir
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
    shuffle_dataset = args.shuffle
    save_iter = args.save_iter
    random_seed = args.seed
    print_terminal = args.print_terminal
    mirrored_img = args.mirrored_imgs
    num_folds = args.num_folds
    # Device Selection (CPU/GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    FLOAT = torch.FloatTensor

    # Define data transformations
    transformations = createTransform(augmentations)
    # Load data
    dataset = PilotNetDataset(path_to_data, mirrored_img, transformations, preprocessing=args.preprocess)
    # Setup KFold cross-validation
    kfold = KFold(n_splits=num_folds, shuffle=shuffle_dataset, random_state=random_seed)

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
        model.classifier[-1] = nn.Linear(num_ftrs, 2)
    elif model_name == 'mobilenet_small':
        if is_pretrained:
            model_weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        model = models.mobilenet_v3_small(weights=model_weights)
        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_ftrs, 2)
    elif model_name == 'resnet':
        if is_pretrained:
            model_weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=model_weights)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
    elif model_name == 'efficientnet_v2':
        if is_pretrained:
            model_weights = models.EfficientNet_V2_S_Weights.DEFAULT
        model = models.efficientnet_v2_s(weights=None)
        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(num_ftrs, 2)
    elif model_name == 'efficientvit':
        model = timm.create_model('efficientvit_b0', pretrained=False)
        num_ftrs = model.head.classifier[-1].in_features
        model.head.classifier[-1] = nn.Linear(num_ftrs, 2)
    elif model_name == 'fastvit':
        model = timm.create_model('fastvit_mci0', pretrained=False)
        num_ftrs = model.head.fc.in_features
        model.head.fc = nn.Linear(num_ftrs, 2)
    model.to(device)
    if os.path.isfile( model_save_dir + '/' + model_name + '_model_{}.pth'.format(random_seed)):
        model.load_state_dict(torch.load(model_save_dir + '/' + model_name + '_model_{}.pth'.format(random_seed),map_location=device,weights_only=True))
        best_model = deepcopy(model)

    # CSV loss log
    self_path = os.getcwd()
    writer_output = csv.writer(open(self_path + '/train_data_' + model_name + '_{}'.format(random_seed) + '.csv', "w"))
    writer_output.writerow(["fold", "epoch", "train loss", "validation loss"])
    
    # Loss and optimizer
    criterion = nn.MSELoss()

    print("*********** Training Started with " +  str(num_folds) + " for Cross-Validation ************")

    # Store cross-validation results
    fold_results = []
    all_train_losses = []
    all_val_losses = []

    # KFold cross-validation training loop
    for fold, (train_idx, val_idx) in enumerate(kfold.split(np.arange(len(dataset)))):
        print(f"\n--- Fold {fold + 1}/{num_folds} ---")
        
        # Create samplers for one fold
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        
        # Create data loaders for one fold
        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler,num_workers=4)
        val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler,num_workers=4)
        
        # Initialize model for one fold
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # Reset best validation loss for one fold
        best_val_loss = float('inf')
        best_fold_model = deepcopy(model)
        
        # Store loss values for one fold
        fold_train_losses = []
        fold_val_losses = []
        
        # Training loop for one fold
        for epoch in range(0, num_epochs):
            model.train()
            train_loss = 0
            global_iter = 0
            
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
                    torch.save(model.state_dict(), model_save_dir + '/' + model_name + '_model_{}_fold{}.pth'.format(random_seed, fold + 1))
                global_iter += 1

                if print_terminal and (i + 1) % 10 == 0:
                    print('Fold [{}/{}], Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                        .format(fold + 1, num_folds, epoch + 1, num_epochs, i + 1, len(train_loader), loss.item()))
            
            # Save epoch train loss
            avg_train_loss = train_loss / len(train_loader)
            fold_train_losses.append(avg_train_loss)
            
            # Validation for one fold
            model.eval()
            with torch.no_grad():
                val_loss = 0 
                for images, labels in val_loader:
                    images = FLOAT(images).to(device)
                    labels = FLOAT(labels.float()).to(device)
                    outputs = model(images)
                    val_loss += criterion(outputs, labels).item()
                    
                avg_val_loss = val_loss / len(val_loader)  # take average
            
            fold_val_losses.append(avg_val_loss)

            # Log loss
            writer_output.writerow([fold + 1, epoch + 1, avg_train_loss, avg_val_loss])

            # Compare and save best model for one fold
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_fold_model = deepcopy(model)
                torch.save(best_fold_model.state_dict(), model_save_dir + '/' + model_name + '_model_best_{}_fold{}.pth'.format(random_seed, fold + 1))
                mssg = "Model Improved!!"
            else:
                mssg = "Not Improved!!"

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print('Fold [{}/{}], Epoch [{}/{}], Train Loss: {:.4f}, Val Loss: {:.4f} {}'.format(
                    fold + 1, num_folds, epoch + 1, num_epochs, avg_train_loss, avg_val_loss, mssg))
        
        # Store results for one fold
        fold_results.append({
            'fold': fold + 1,
            'best_val_loss': best_val_loss,
            'train_losses': fold_train_losses,
            'val_losses': fold_val_losses,
            'best_model': best_fold_model
        })
        all_train_losses.append(fold_train_losses)
        all_val_losses.append(fold_val_losses)
        
        print(f"Fold {fold + 1} completed. Best Validation Loss: {best_val_loss:.4f}\n")
        

    # Select best model from all folds
    best_fold = min(fold_results, key=lambda x: x['best_val_loss'])
    print(f"\nBest model from Fold {best_fold['fold']} with validation loss: {best_fold['best_val_loss']:.4f}")
    model = best_fold['best_model']
    
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
    
    print(f'Test loss: {test_loss/len(test_loader)}')

    # Plot cross-validation results
    plt.figure(figsize=(12, 5))
    
    # Plot average training and validation losses across folds
    plt.subplot(1, 2, 1)
    avg_train_per_epoch = [np.mean([all_train_losses[fold][epoch] for fold in range(num_folds)]) 
                           for epoch in range(num_epochs)]
    avg_val_per_epoch = [np.mean([all_val_losses[fold][epoch] for fold in range(num_folds)]) 
                         for epoch in range(num_epochs)]
    
    plt.plot(range(1, num_epochs + 1), avg_train_per_epoch, marker='o', label="Train (avg)", linewidth=2)
    plt.plot(range(1, num_epochs + 1), avg_val_per_epoch, marker='o', label="Val (avg)", linewidth=2)
    plt.title('Average Loss Across Folds')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend(loc="upper left")
    
    # Plot individual fold results
    plt.subplot(1, 2, 2)
    for fold in range(num_folds):
        plt.plot(range(1, num_epochs + 1), all_val_losses[fold], marker='o', 
                label=f"Fold {fold + 1}", alpha=0.7)
    plt.title('Validation Loss per Fold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend(loc="upper left")
    
    plt.tight_layout()
    plt.show()
    
    # Print cross-validation summary
    print("\n========== Cross-Validation Summary ==========")
    for result in fold_results:
        print(f"Fold {result['fold']}: Best Val Loss = {result['best_val_loss']:.4f}")
    
