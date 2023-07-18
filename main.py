#!/usr/bin/env python3
import argparse
import logging
import time
import numpy as np
from waggle.plugin import Plugin
from waggle.data.vision import Camera
from __future__ import print_function, division
import os
from PIL import Image
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from PIL import Image, ImageChops
import io
plt.ion()



# models setup

#vgg16 setup
vgg16 = models.vgg16_bn()

for param in vgg16.features.parameters():
    param.require_grad = False
class_names = ["cloud","other", "smoke"]

num_features = vgg16.classifier[6].in_features
features = list(vgg16.classifier.children())[:-1] # Remove last layer
features.extend([nn.Linear(num_features, len(class_names))]) # Add our layer with 4 outputs
vgg16.classifier = nn.Sequential(*features) # Replace the model classifier


#resnet18 setup
resnet18 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
num_features = resnet18.fc.in_features     #extract fc layers features
resnet18.fc = nn.Linear(num_features, 3) #(num_of_class == 2)

#resnet34 setup
resnet34 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True)
num_features = resnet34.fc.in_features     #extract fc layers features
resnet34.fc = nn.Linear(num_features, 3) #(num_of_class == 2)

#resnet50 setup
resnet50 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
num_features = resnet50.fc.in_features     #extract fc layers features
resnet50.fc = nn.Linear(num_features, 3) #(num_of_class == 2)





#Gets Prediction of Tile
def get_prediction(model, image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    output = model.forward(tensor)
     
    probs = torch.nn.functional.softmax(output, dim=1)
    conf, classes = torch.max(probs, 1)
    print(classes.item())
    return conf.item(), class_names[classes.item()]

#Transforms Image Into Tensor
def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)

#performs inference on image
def ImageInference(model = model, image = sample):

    fullimage = Image.open(image)
    fullimage = fullimage.resize((1344, 1344))
    fullimage = fullimage.crop((0, 448, 1344, 1344))

    data = []
    for i in range(6):
        for k in range(4):
            model.load_state_dict(torch.load('/home/jszumny/Documents/copy of smoke data/newnew/attempts/invert/resnet50model.pt',map_location=torch.device('cpu')))

            model.eval()

            tile = fullimage.crop((i*224, k*224, (i+1)*224, (k+1)*224))
            # tile = ImageChops.invert(tile)
            tile.save('/home/jszumny/Documents/test/' + str(i) + "_" + str(k) + '.jpg')
            
            image_path= ("/home/jszumny/Documents/test/" + str(i) + "_" + str(k) + '.jpg')
            image = plt.imread(image_path)
            plt.imshow(image)
            
            with open(image_path, 'rb') as f:
                image_bytes = f.read()
                
                conf,y_pre=get_prediction(model = model, image_bytes=image_bytes)
                print(y_pre, ' at confidence score:{0:.2f}'.format(conf))
        
            d = {"xtile": str(i), "ytile": str(k),"class": y_pre, "percentage": '{0:.2f}'.format(conf)}
            data.append(d)
            
    return data



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=0, help="camera device to use")
    parser.add_argument("--interval", default=10, type=float, help="sampling interval in seconds")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S')

    logging.info("starting plugin. will process a frame every %ss", args.interval)

    with Plugin() as plugin:
        cam = Camera(args.device)

        for sample in cam.stream():
            logging.info("processing frame")
            results = ImageInference(sample.data)
            
            logging.info("image inference")
            logging.info("data: %s", results["data"])

            plugin.publish("image.data", results["data"])
           

            logging.info("published summary")

            time.sleep(args.interval)


if __name__ == "__main__":
    main()