import math
import os
import torch.nn as nn
import torch.utils
import torch.distributions
import torchvision
from sklearn.model_selection import train_test_split

import autoencoder
import  classifier
IMGSIZE=64
latent_dims = 10
criterion = nn.MSELoss()

train_path = os.getcwd() + "\\images2"
transform_d = torchvision.transforms.Compose([torchvision.transforms.CenterCrop(IMGSIZE),
                                              torchvision.transforms.Grayscale(1),
                                              torchvision.transforms.ToTensor()])
all_data = torchvision.datasets.ImageFolder(train_path, transform=transform_d)
division = [int(math.floor(len(all_data) * 0.8)), int(math.ceil(len(all_data) * 0.2))]
trainData, valData = torch.utils.data.random_split(all_data, division)
train_loader = torch.utils.data.DataLoader(trainData, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(valData, batch_size=64, shuffle=True)

autoencoderModel = autoencoder.Autoencoder(latent_dims)
autoencoderModel = autoencoder.train(autoencoderModel, train_loader) #takes some time
tot_loss=autoencoder.val(autoencoderModel, test_loader)
print(tot_loss)





all_data = torchvision.datasets.ImageFolder(train_path, transform=transform_d)
decoded_imgs,labels=classifier.use_encoder(all_data,autoencoderModel)
trainData, valData = torch.utils.data.random_split([(decoded_imgs[i], labels[i]) for i in range(len(all_data))], division)
train_loader = torch.utils.data.DataLoader(trainData, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(valData, batch_size=64, shuffle=True)
DNN_model = classifier.DNN()
DNN_model=classifier.DNN_train(DNN_model, train_loader)
tot_loss=classifier.DNN_val(DNN_model, test_loader)
print('test loss: ',tot_loss)




