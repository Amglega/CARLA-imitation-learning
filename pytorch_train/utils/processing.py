import os
import glob
import numpy as np
import cv2
from tqdm import tqdm
import csv
import matplotlib.pyplot as plt


def load_data(folder):

    name_folder = folder
    # Remove any existing trailing slashes
    while name_folder.endswith('/'):
        name_folder = name_folder[:-1]
    # Add a single trailing slash for the path
    name_folder = name_folder + '/'
    list_images = glob.glob(name_folder + '*.png')
    images = sorted(list_images, key=lambda x: int(x.split('/')[-1].split('.png')[0]))
    name_file = os.path.join(folder,'data.csv') #'/data.json'
    file = open(name_file, 'r')
    reader = csv.DictReader(file)
    data = []
    for row in reader: # reading all values
        data.append((row['throttle'], row['steer']))
    file.close()
    return images, data

def get_images(list_images, type_image, array_imgs):
    # Read the images
    for name in tqdm(list_images):
        img = cv2.imread(name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if type_image == 'cropped':
            img = img[240:480, 0:640]
            img = cv2.resize(img, (int(200), int(66)))
        else:
            target_height = int(66)
            target_width = int(target_height * img.shape[1]/img.shape[0])
            img_resized = cv2.resize(img, (target_width, target_height))
            padding_left = int((200 - target_width)/2)
            padding_right = 200 - target_width - padding_left
            img = cv2.copyMakeBorder(img_resized.copy(),0,0,padding_left,padding_right,cv2.BORDER_CONSTANT,value=[0, 0, 0])
            
        array_imgs.append(img)

    # Display the loaded image
    #plt.imshow(cropped_img_rgb)
    #plt.axis('off') # Hide axes
    #plt.title('Dataset Image')
    #plt.show()

    return array_imgs

def parse_json(data, array):
    # Process json
    data_parse = data.split('}')[:-1]
    for d in data_parse:
        v = d.split('"v": ')[1]
        d_parse = d.split(', "v":')[0]
        w = d_parse.split(('"w": '))[1]
        array.append((float(v), float(w)))

    return array

def parse_csv(data, array):
    # Process csv

    for v, w in data:
        array.append((float(v), float(w)))
    return array


def preprocess_data(array, imgs, flip, data_type):
    
    new_array = array
    new_array_imgs = imgs

    # Data augmentation

    if flip:
        # Take the image and just flip it and negate the measurement
        flip_imgs = []
        array_flip = []
        for i in tqdm(range(len(imgs))):
            flip_imgs.append(cv2.flip(imgs[i], 1))
            array_flip.append((array[i][0], -array[i][1]))
        new_array += array_flip
        new_array_imgs += flip_imgs
    
    if data_type == 'extreme':
        extreme_case_1_img = []
        extreme_case_2_img = []
        extreme_case_1_array = []
        extreme_case_2_array = []

        for i in tqdm(range(len(new_array_imgs))):
            if abs(new_array[i][1]) > 2:
                extreme_case_2_img.append(new_array_imgs[i])
                extreme_case_2_array.append(new_array[i])
            elif abs(new_array[i][1]) > 1:
                extreme_case_1_img.append(new_array_imgs[i])
                extreme_case_1_array.append(new_array[i])

        new_array += extreme_case_1_array*5 + extreme_case_2_array*10
        new_array_imgs += extreme_case_1_img*5 + extreme_case_2_img*10

    return new_array, new_array_imgs



def check_path(path):
    if not os.path.exists(path):
        print(f"{path} not exist")
        os.makedirs(path)
        print(f"Create {path} success")
