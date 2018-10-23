import ssd_net
import mobilenet
import torch.nn.functional as F
import bbox_loss
import cityscape_dataset
import bbox_helper
import util.module_util
import os
from glob import glob
import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import torch
import matplotlib.patches as patches

import random
import torch.optim
from torch.autograd import Variable


class Training():
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
            #print(image_path)
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
            train_datalist.append({'image_path': image_path.rstrip(), 'labels': labels, 'bound_boxes': bound_boxes})

        random.shuffle(train_datalist)
        #print(train_datalist)
        n_train_sets = 0.8 * len(train_datalist)

        train_sets = train_datalist[: int(n_train_sets)]

        im = np.array(Image.open(train_sets[1]['image_path']), dtype=np.uint8)

        # Create figure and axes
        fig, ax = plt.subplots(1)

        # Display the image
        ax.imshow(im)

        # Create a Rectangle patch
        for i in range(0,len(train_sets[1]['labels'])):
            if train_sets[1]['labels'][i] != 0:
                rect = patches.Rectangle((train_sets[1]['bound_boxes'][i][0],train_sets[1]['bound_boxes'][i][1]),train_sets[1]['bound_boxes'][i][2],train_sets[1]['bound_boxes'][i][3], linewidth=1, edgecolor='r', facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)

        plt.show()



        train_dataset = cityscape_dataset.CityScapeDataset(train_sets)
        train_data_loader: object = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=20,
                                                        shuffle=True,
                                                        num_workers=0)
        print('Total training items', len(train_dataset), ', Total training mini-batches in one epoch:',
              len(train_data_loader))

        n_valid_sets = 0.2 * len(train_datalist)
        valid_sets= train_datalist[int(n_train_sets): int(n_train_sets + n_valid_sets)]

        valid_set = cityscape_dataset.CityScapeDataset(valid_sets)
        valid_data_loader = torch.utils.data.DataLoader(valid_set,
                                                        batch_size=20,
                                                        shuffle=True,
                                                        num_workers=0)
        print('Total validation set:', len(valid_set), ', Total training mini-batches in one epoch:',
              len(valid_data_loader))

        ssd = ssd_net.SSD().cuda()
        #print(ssd)

        criterion = bbox_loss.MultiboxLoss((0.1,0.2))


        optimizer = torch.optim.Adam(ssd.classifier.parameters(), lr=0.01)

        train_losses = []

        max_epochs = 1
        itr = 0
        #print(train_data_loader)
        for epoch_idx in range(0, max_epochs):
            for img_tensor, train_input, train_label in train_data_loader:

                itr += 1

                # Set the network works in GPU
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                ssd = ssd.to(device)
                # print(device)
                for img,features, targets in train_data_loader:
                    features = features.to(device)
                    targets = targets.to(device)

                # switch to train model
                ssd.train()

                # zero the parameter gradients
                optimizer.zero_grad()

                # Forward

                train_input = Variable(img_tensor.cuda())  # use Variable(*) to allow gradient flow
                #print(train_input.dim())
                confidence , train_out = ssd.forward(train_input)  # forward once
                #print(train_out)
                # compute loss
                train_label = Variable(train_label.cuda().float())
                loss = criterion.forward(confidence,train_out,train_label,train_input)

                # do the backward and compute gradients
                loss.backward()
                #
                # update the parameters with SGD
                optimizer.step()

                train_losses.append((itr,loss.item()))

                if itr % 200 == 0:
                    print('Epoch: %d Itr: %d Loss: %f' % (epoch_idx, itr, loss.item()))

        train_losses = np.asarray(train_losses)
        # plt.plot(itr,      # iteration
        #         loss.item())      # loss value
        # plt.show()

