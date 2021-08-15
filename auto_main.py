import math
import os
import torch.nn as nn
import torch.utils
import torch.distributions
import torchvision
import autoencoder
import  classifier

IMGSIZE=64
latent_dims = 5
criterion = nn.MSELoss()

## import data and maintain it in train/test loaders
train_path = os.getcwd() + "\\images2"
transform_d = torchvision.transforms.Compose([torchvision.transforms.CenterCrop(IMGSIZE),
                                              torchvision.transforms.Grayscale(1),
                                              torchvision.transforms.ToTensor()])

all_data = torchvision.datasets.ImageFolder(train_path, transform=transform_d)
#divide data to train and test
division = [int(math.floor(len(all_data) * 0.8)), int(math.ceil(len(all_data) * 0.2))]
trainData, valData = torch.utils.data.random_split(all_data, division)
train_loader = torch.utils.data.DataLoader(trainData, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(valData, batch_size=64, shuffle=True)

#create autoencoder model
autoencoderModel = autoencoder.Autoencoder(latent_dims)
#train the model
autoencoderModel = autoencoder.train(autoencoderModel, train_loader)
#test the model
tot_loss=autoencoder.val(autoencoderModel, test_loader)
print('test loss: ',tot_loss)



## now we do the same thing but we use the encoded vector from autoencoder
all_data = torchvision.datasets.ImageFolder(train_path, transform=transform_d)
decoded_imgs,labels=classifier.use_encoder(all_data,autoencoderModel)
trainData, valData = torch.utils.data.random_split([(decoded_imgs[i], labels[i]) for i in range(len(all_data))], division)
train_loader = torch.utils.data.DataLoader(trainData, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(valData, batch_size=64, shuffle=True)

DNN_model = classifier.DNN()
DNN_model=classifier.DNN_train(DNN_model, train_loader)
tot_loss=classifier.DNN_val(DNN_model, test_loader)
print('test loss: ',tot_loss)




