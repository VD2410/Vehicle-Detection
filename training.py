import ssd_net
import mobilenet
import bbox_loss
import cityscape_dataset
import bbox_helper
import module_util
import os
from glob import glob
import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import torch
import random
import torch.optim
from torch.autograd import Variable

# Set default tenosr type, 'torch.cuda.FloatTensor' is the GPU based FloatTensor
torch.set_default_tensor_type('torch.cuda.FloatTensor')

current_directory = os.getcwd()  # current working directory
training_ratio = 0.8

if __name__ == '__main__':
    poly_init_fol_name = "cityscapes_samples_labels"
    init_img_path = "cityscapes_samples"

    compl_poly_path = os.path.join(current_directory, poly_init_fol_name, "*", "*_polygons.json")

    polygon_folder = glob(compl_poly_path)

    polygon_folder = np.array(polygon_folder)

    img_label_list = []

    for file in polygon_folder:
        with open(file, "r") as f:
            frame_info = json.load(f)
            obj_length = len(frame_info['objects'])
            file_path = file
            image_name = file_path.split("/")[-1][:-23]
            for i in range(obj_length):
                label = frame_info['objects'][i]['label']
                if label == "ego vehicle":
                    break
                polygon = np.array(frame_info['objects'][i]['polygon'], dtype=np.float32)
                left_top = np.min(polygon, axis=0)
                right_bottom = np.max(polygon, axis=0)
                concat = np.concatenate((left_top, right_bottom))
                img_label_list.append(
                    {'image_name': image_name, 'file_path': file_path, 'label': label, 'bbox': concat})

    label_length = len(img_label_list)

    # get images list

    img_path = os.path.join(init_img_path, "*", "*")

    images = glob(img_path)

    images = np.array(images)


    train_datalist = []
    for i in range(0, len(images)):
        img_folder = images[i].split('/')[-2]
        img_name = images[i].split('/')[-1]
        img_class = img_name[:-16]
        image_path = os.path.join(init_img_path, img_folder, img_name)
        print(image_path)
        bound_boxes = []
        labels = []
        for i in range(label_length):
            if img_label_list[i]["image_name"] == img_class:
                bbox = img_label_list[i]['bbox']
                bound_boxes.append(bbox)
                if img_label_list[i]['label'] in ('car', 'cargroup'):
                    label = 1
                elif img_label_list[i]['label'] in ('person', 'persongroup'):
                    label = 2
                elif img_label_list[i]['label'] == 'traffic sign':
                    label = 3
                else:
                    label = 0
                labels.append(label)
        train_datalist.append({'image_path': image_path, 'labels': labels, 'bound_boxes': bound_boxes})

    random.shuffle(train_datalist)
    train_validate_items = len(train_datalist)
    print(train_validate_items)