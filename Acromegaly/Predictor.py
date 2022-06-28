import os
from PIL import Image
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, utils
import numpy as np

from flask import Flask, render_template, request

app = Flask(__name__)

class BasicBlock(nn.Module):
    
    def __init__(self, number_of_input_channels,  number_of_output_channels, stride = 1, downsample = None):
        
        super().__init__()
        
        self.first_convolution = nn.Conv2d(number_of_input_channels, 
                                           number_of_output_channels, 
                                           padding = 1, 
                                           kernel_size = 3, 
                                           stride = stride, 
                                           bias = False)
        self.batch_normalization_layer_one = nn.BatchNorm2d(number_of_output_channels)
        self.first_relu = nn.ReLU(inplace = True)
        
        self.second_convolution = nn.Conv2d(number_of_output_channels, 
                                            number_of_output_channels, 
                                            padding = 1, 
                                            kernel_size = 3, 
                                            stride = 1, 
                                            bias = False)
        self.batch_normalization_layer_two = nn.BatchNorm2d(number_of_output_channels)
        self.second_relu = nn.ReLU(inplace = True)
        
        self.downsample = downsample
        
    ########################################################################################################################### 
    
    def forward(self, input_tensor):
        
        identity = input_tensor

        output_tensor = self.first_convolution(input_tensor)
        output_tensor = self.batch_normalization_layer_one(output_tensor)
        output_tensor = self.first_relu(output_tensor)

        output_tensor = self.second_convolution(output_tensor)
        output_tensor = self.batch_normalization_layer_two(output_tensor)

        if self.downsample is not None:
            identity = self.downsample(input_tensor)

        output_tensor = output_tensor + identity
        output_tensor = self.second_relu(output_tensor)
        
        return output_tensor
    
###############################################################################################################################################################################

class ResNet34(nn.Module):

    def __init__(self, number_of_output_classes = 2):
        
        super().__init__()
        
        self.number_of_input_channels = 64
        
        self.layer_one = self.make_layer_one()
        self.layer_two = self.make_layer(64, 3)
        self.layer_three = self.make_layer(128, 4)
        self.layer_four = self.make_layer(256, 6)
        self.layer_five = self.make_layer(512, 3, stride = 1)
        
        self.average_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fully_connected_layer = nn.Linear(512 , number_of_output_classes)


    def make_layer(self, number_of_output_channels, number_of_blocks, stride = 2):
        
        downsample = None
        
        if stride != 1 or self.number_of_input_channels != number_of_output_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.number_of_input_channels, number_of_output_channels, kernel_size = 1, stride = stride, bias = False),
                nn.BatchNorm2d(number_of_output_channels),
            )

        blocks = list()
        first_block = BasicBlock(self.number_of_input_channels, number_of_output_channels, stride = stride, downsample = downsample)
        blocks.append(first_block)
        
        self.number_of_input_channels = number_of_output_channels
        
        for _ in range(1, number_of_blocks):
            block = BasicBlock(self.number_of_input_channels, number_of_output_channels)
            blocks.append(block)

        return nn.Sequential(*blocks)
    
    def make_layer_one(self):
        
        convolution_layer_one = nn.Conv2d(3, self.number_of_input_channels, kernel_size = 7, stride = 2, padding = 3, bias=False)
        batch_normalisation_layer_one = nn.BatchNorm2d(self.number_of_input_channels)
        relu_layer_one = nn.ReLU(inplace=True)
        maxpool_layer_one = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        components = list()
        
        components.append(convolution_layer_one)
        components.append(batch_normalisation_layer_one)
        components.append(relu_layer_one)
        components.append(maxpool_layer_one)
        
        return nn.Sequential(*components)
        
    def forward(self, input_tensor):
        
        transforming_tensor = self.layer_one(input_tensor)
        transforming_tensor = self.layer_two(transforming_tensor)    
        transforming_tensor = self.layer_three(transforming_tensor)         
        transforming_tensor = self.layer_four(transforming_tensor)         
        transforming_tensor = self.layer_five(transforming_tensor)          
        transforming_tensor = self.average_pool(transforming_tensor)
        transforming_tensor = torch.flatten(transforming_tensor, 1)
        output_tensor_logits = self.fully_connected_layer(transforming_tensor)
        output_tensor_probabilities = F.softmax(output_tensor_logits, dim = 1)
        
        return output_tensor_logits, output_tensor_probabilities
    
###############################################################################################################################################################################

@app.route("/", methods = ['POST'])
def predict():
    if request.method == 'POST':
        image = request.files['hand']
        filename = request.files['hand'].filename
        image = Image.open(image).convert('RGB')
        resize = transforms.Resize([448, 448])
        resized_image = resize(image)
        to_tensor = transforms.ToTensor()
        input_tensor = to_tensor(resized_image)
        input_tensor = input_tensor.unsqueeze(0)

        weights = torch.load("C:/Users/suhas/OneDrive/Desktop/Acromegaly/SavedModels/BestModel.pkl")
        model = ResNet34()
        model.load_state_dict(weights)

        logits, probabilities = model(input_tensor)
        probabilities = probabilities.tolist()
        probability_of_acromegaly = round(probabilities[0][0], 4)
        probability_of_no_acromegaly = round(1 - probability_of_acromegaly, 4)

        reader = open("Best_Values.txt", "r")
        best_sensitivity = reader.readline()
        best_specificity = reader.readline()
        best_ppv = reader.readline()
        best_npv = reader.readline()
        best_f1_score = reader.readline()
        best_accuracy = reader.readline()

        return render_template('Acromegaly.html', 
        filename = filename, 
        probability_of_acromegaly = str(probability_of_acromegaly), 
        probability_of_no_acromegaly = str(probability_of_no_acromegaly),
        best_sensitivity = str(best_sensitivity),
        best_specificity = str(best_specificity),
        best_ppv = str(best_ppv),
        best_npv = str(best_npv),
        best_f1_score = str(best_f1_score),
        best_accuracy = str(best_accuracy))

if __name__ == '__main__':
    app.run(host='localhost', port=5000, debug=True)
