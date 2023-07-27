from __future__ import print_function, division
#!/usr/bin/env python3
import argparse
import logging
import time
import numpy as np
from waggle.plugin import Plugin
from waggle.data.vision import Camera
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
import cv2
from matplotlib import cm
plt.ion()


# models setup
def get_arguments():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model on live images."
    )

    # parser.add_argument('--weight', type=str, default='vgg16.pt', help='model name')
    parser.add_argument('--labels', dest='labels',
                        action='store', default='classes', type=str,
                        help='Labels for detection')


    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')


    parser.add_argument(
        '-stream', dest='stream',
        action='store', default="XNV-8081Z",
        help='ID or name of a stream, e.g. sample')
    parser.add_argument(
        '-continuous', dest='continuous',
        action='store_true', default=False,
        help='Continuous run flag')
    parser.add_argument(
        '-sampling-interval', dest='sampling_interval',
        action='store', default=-1, type=int,
        help='Sampling interval between inferencing')


    return parser.parse_args()


def load_class_names(namesfile):
    return ["cloud",'other', 'smoke']

class VGG16():
    def __init__(self, args, weightfile):

        self.use_cuda = torch.cuda.is_available()

        if self.use_cuda:
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        logging.info("ini class")
        self.model = models.vgg16(weightfile)
        
        logging.info("model is in")
        
        self.model = self.model.half()
        self.model.eval()
        self.class_names = load_class_names(args.labels)


    def run(self, tile):

       

        image = tile / 255.0
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image).to(self.device).half()
        image = image.unsqueeze(0)

        with torch.no_grad():
            pred = self.model(image)[0]
    
        return pred, self.class_names

#vgg16 setup
vgg16 = models.vgg16_bn()

for param in vgg16.features.parameters():
    param.require_grad = False
class_names = ["cloud","other", "smoke"]

num_features = vgg16.classifier[6].in_features
features = list(vgg16.classifier.children())[:-1] # Remove last layer
features.extend([nn.Linear(num_features, len(class_names))]) # Add our layer with 4 outputs
vgg16.classifier = nn.Sequential(*features) # Replace the model classifier


# #resnet18 setup
# resnet18 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
# num_features = resnet18.fc.in_features     #extract fc layers features
# resnet18.fc = nn.Linear(num_features, 3) #(num_of_class == 2)

# #resnet34 setup
# resnet34 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True)
# num_features = resnet34.fc.in_features     #extract fc layers features
# resnet34.fc = nn.Linear(num_features, 3) #(num_of_class == 2)

# #resnet50 setup
# resnet50 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
# num_features = resnet50.fc.in_features     #extract fc layers features
# resnet50.fc = nn.Linear(num_features, 3) #(num_of_class == 2)





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
def ImageInference(vgg16, image):
    logging.info("Resizing and Cropping Image")
    fullimage = cv2.resize(image,(1344, 1344))
    fullimage = fullimage[448:1344, 0:1344]
    logging.info("Looping through tiles")
    data = []
    for i in range(6):
        for k in range(4):
            
            tile = fullimage[(i*224):((i+1)*224), (k*224):((k+1)*224)]
            pred, conf = vgg16.run(tile=tile)
            
            d = {"xtile": str(i), "ytile": str(k),"class": pred, "percentage": '{0:.2f}'.format(conf)}
            data.append(d)
    
    df = pd.DataFrame(data)
    return df



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="bottom_camera", help="camera device to use")
    parser.add_argument("--interval", default=10, type=float, help="sampling interval in seconds")
    parser.add_argument('--weight', type=str, default='vgg16.pt', help='model name')
    args = parser.parse_args()

    

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S')

    logging.info("loading model weights")

    vgg16 = VGG16(args, args.weight)

    logging.info("model weights loaded")


    logging.info("starting plugin. will process a frame every %ss", args.interval)

    with Plugin() as plugin:
        cam = Camera(args.device)

        for sample in cam.stream():
            logging.info("processing frame")
            image = sample.data

            logging.info("grabbed image")
            results = ImageInference(vgg16, image)

            # print(sample.data)
            results.to_csv("results.csv")
            logging.info("image inference")
            logging.info("data: %s", results["data"])
        
            plugin.upload_file("results.csv")

            plugin.publish("smoke_detection", len(results[results["class"] == "smoke"]))
           
            logging.info("published summary")

            time.sleep(args.interval)


if __name__ == "__main__":
    main() 
main()