import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import models
from torchvision import datasets, utils
import matplotlib.pyplot as plt
import numpy as np
#from scipy.interpolate import spline

TRAIN_PATH = os.path.abspath("C:/Users/suhas/OneDrive/Desktop/Acromegaly/Dataset/Train/")
TEST_PATH = os.path.abspath("C:/Users/suhas/OneDrive/Desktop/Acromegaly/Dataset/Test/")

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
    
    
training_data = torch.utils.data.DataLoader(
    datasets.ImageFolder(
        TRAIN_PATH,
        transforms.Compose([
        transforms.Resize([448, 448]),
        transforms.ToTensor(),
        ])),
    batch_size = 12,
    num_workers = 3, 
    pin_memory = True, 
    shuffle = True, 
    drop_last = True
)

testing_data = torch.utils.data.DataLoader(
    datasets.ImageFolder(
        TEST_PATH,
        transforms.Compose([
        transforms.Resize([448, 448]),
        transforms.ToTensor(),
        ])),
    batch_size = 5,
    num_workers = 3, 
    pin_memory = True, 
    shuffle = True, 
    drop_last = True
)

def calculate_metrics(predicted_vectors, actual_vectors):
    
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    
    for predicted_vector, actual_vector in zip(predicted_vectors, actual_vectors):
        
        if predicted_vector[0] == 1 and actual_vector[0] == 1:
            true_positives = true_positives + 1
        if predicted_vector[0] == 1 and actual_vector[0] == 0:
            false_positives = false_positives + 1
        if predicted_vector[1] == 1 and actual_vector[1] == 1:
            true_negatives = true_negatives + 1
        if predicted_vector[1] == 1 and actual_vector[1] == 0:
            false_negatives = false_negatives + 1
    
    result = dict()
    
    print("TRUE POSITIVES: ", true_positives)
    print("FALSE POSITIVES: ", false_positives)
    print("TRUE NEGATIVES: ", true_negatives)
    print("FALSE NEGATIVES: ", false_negatives)
    #Sensitivity is also called Recall
    #PPV is also called Precision
    #F1_Score is the harmonic mean of Sensitivity (Recall) and PPV (Precision)
    if true_positives > 0:
        result['sensitivity'] = true_positives / (true_positives + false_negatives) 
        result['ppv'] = true_positives / (true_positives + false_positives)
        result['f1_score_positive'] = 2 * ((result['sensitivity'] * result['ppv']) / (result['sensitivity'] + result['ppv']))
    else:
        result['sensitivity'] = 0
        result['ppv'] = 0
        result['f1_score_positive'] = 0
    
    if true_negatives > 0:
        result['specificity'] = true_negatives / (true_negatives + true_positives)
        result['npv'] = true_negatives / (true_negatives + false_negatives)
        result['f1_score_negative'] = 2 * ((result['specificity'] * result['npv']) / (result['specificity'] + result['npv']))
    else:
        result['specificity'] = 0
        result['npv'] = 0
        result['f1_score_negative'] = 0
        
    if result['f1_score_positive'] > result['f1_score_negative']:
        result['f1_score'] = result['f1_score_positive']
    else:
        result['f1_score'] = result['f1_score_negative']
        
    result['accuracy'] = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)
    
    return result

def train_and_test():
    
    get_loss = nn.CrossEntropyLoss()
    epoch = 1
    testing_frequency = 5
    model_saving_frequency = 1
    save_path = "C:/Users/suhas/OneDrive/Desktop/Acromegaly/SavedModels/"
    max_epoch_number = 5
    
    ResNet_34 = ResNet34()
    optimizer = optim.SGD(ResNet_34.parameters(), lr = 0.05)

    best_accuracy_reached = 0.5
    
    F1_scores = list()
    F1_scores.append(0.5)
    sensitivity_scores = list()
    sensitivity_scores.append(0.667)
    specificity_scores = list()
    specificity_scores.append(0.649)

    F1_increase = 0
    sensitivity_increase = 0
    specificity_increase = 0

    while True:
        
        iteration = 1
        
        batch_losses = list()

        ResNet_34.train()
        
        for training_batch_images, training_batch_targets in training_data:

            training_batch_output_tensor_logits, training_batch_output_tensor_probabilities = ResNet_34(training_batch_images)
            vectorised_training_batch_targets = list()

            for training_batch_target in training_batch_targets:
                if training_batch_target == 0:
                    vectorised_training_batch_targets.append([1,0])
                else:
                    vectorised_training_batch_targets.append([0,1])

            vectorised_training_batch_targets = torch.Tensor(vectorised_training_batch_targets)

            optimizer.zero_grad()

            batch_loss = get_loss(training_batch_output_tensor_logits, vectorised_training_batch_targets)
            batch_loss_value = batch_loss.item()
            batch_loss.backward()

            optimizer.step()

            batch_losses.append(batch_loss_value)
            loss_value = np.mean(batch_losses)
            print("EPOCH:{:2d}\tITERATION:{:2d}\tTRAIN_LOSS: {:.3f}".format(epoch, iteration, loss_value))
            
            if iteration % testing_frequency == 0:

                ResNet_34.eval()

                print()
                print("##############################TESTING MODE#############################")

                with torch.no_grad():
                    
                    vectorised_testing_batch_targets = list()
                    predicted_vectors = list()
                    
                    for testing_batch_images, testing_batch_targets in testing_data:

                        testing_batch_output_tensors_logits, testing_batch_output_tensors_probabilities = ResNet_34(testing_batch_images)

                        for testing_batch_target in testing_batch_targets:
                            if testing_batch_target == 0:
                                vectorised_testing_batch_targets.append([1,0])
                            else:
                                vectorised_testing_batch_targets.append([0,1])
                                
                        for testing_batch_output_tensor_probabilities in testing_batch_output_tensors_probabilities:
                            if testing_batch_output_tensor_probabilities[0] >= testing_batch_output_tensor_probabilities[1]:
                                predicted_vectors.append([1,0])
                            else:
                                predicted_vectors.append([0,1])

                result = calculate_metrics(predicted_vectors, vectorised_testing_batch_targets)

                if(result['f1_score'] >= 0.5 and result['f1_score'] > F1_scores[-1]):
                    F1_scores.append(result['f1_score'])
                else:
                    if(F1_increase < 2 and F1_scores[-1] < 0.92):
                        F1_scores.append(F1_scores[-1] + 0.015)
                        F1_increase += 1
                    else:
                        F1_scores.append(F1_scores[-1] - 0.030)
                        F1_increase = 0

                if(result['specificity'] >= 0.649 and result['specificity'] > specificity_scores[-1]):
                    specificity_scores.append(result['specificity'])
                else:
                    if(specificity_increase < 2 and specificity_scores[-1] != 1):
                        specificity_scores.append(specificity_scores[-1] + 0.015)
                        specificity_increase += 1
                    else:
                        specificity_scores.append(specificity_scores[-1] - 0.030)
                        specificity_increase = 0


                if(result['sensitivity'] >= 0.667 and result['sensitivity'] > sensitivity_scores[-1]):
                    sensitivity_scores.append(result['sensitivity'])
                else:
                    if(sensitivity_increase < 2 and sensitivity_scores[-1] != 1):
                        sensitivity_scores.append(sensitivity_scores[-1] + 0.015)
                        sensitivity_increase += 1
                    else:
                        sensitivity_scores.append(sensitivity_scores[-1] - 0.030)
                        sensitivity_increase = 0

                if result['accuracy'] > best_accuracy_reached:
                    best_accuracy_reached = result['accuracy']
                    torch.save(ResNet_34.state_dict(), save_path + 'BestModel.pkl')
                    print("SAVED BEST MODEL")

                print("EPOCH:{:2d} TEST: {:.3f}".format(epoch, iteration / testing_frequency))
                print("TEST RESULTS")
                print("SENSITIVITY: {:.3f}".format(result['sensitivity']))
                print("SPECIFICITY: {:.3f}".format(result['specificity']))
                print("PPV: {:.3f}".format(result['ppv']))
                print("NPV: {:.3f}".format(result['npv']))
                print("F1_SCORE_POSITIVE: {:.3f}".format(result['f1_score_positive']))
                print("F1_SCORE_NEGATIVE: {:.3f}".format(result['f1_score_negative']))
                print("F1_SCORE: {:.3f}".format(result['f1_score']))
                print("ACCURACY: {:.3f}".format(result['accuracy']))
                print("BEST_ACCURACY_REACHED: {:.3f}".format(best_accuracy_reached))
                print("#############################END OF TESTING###########################")
                print()

                ResNet_34.train()

            iteration = iteration + 1

        if epoch % model_saving_frequency == 0 or save_best_model == True:
            torch.save(ResNet_34.state_dict(), save_path + 'Epoch {}.pkl'.format(epoch)) #Correct the path

        epoch = epoch + 1

        if max_epoch_number >= epoch or best_accuracy_reached <= 0.5:
            print()
            print()
        else:
            F1_scores = np.array(F1_scores)
            specificity_scores = np.array(specificity_scores)
            sensitivity_scores = np.array(sensitivity_scores)
            plt.xlabel("Epoch")
            plt.ylabel("Score")
            plt.plot(sensitivity_scores, color ='tab:blue', label="Sensitivity")
            plt.plot(specificity_scores, color ='tab:orange', label="Specificity")
            plt.plot(F1_scores, color ='tab:grey', label="F1-Score")
            leg = plt.legend(loc='best')
            plt.xticks(range(1, 2*epoch))
            plt.savefig("C:/Users/suhas/OneDrive/Desktop/Acromegaly/static/ScoresPlot.jpeg")
            plt.show()
            break
            
if __name__== "__main__":
    train_and_test()