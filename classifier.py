import torch.utils
import torch.distributions
import math
import os
import torch
import torch.nn as nn
import torch.utils
import torch.distributions
import torchvision
import torch.nn.functional as F
from sklearn import metrics
from sklearn.metrics import plot_roc_curve, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_curve, auc, roc_auc_score

latent_dims=10
IMGSIZE=64
criterion = nn.MSELoss()
target = ['glioma', 'meningioma', 'no-tumor', 'pituitary']

# set plot figure size
fig, c_ax = plt.subplots(1,1, figsize = (12, 8))

# function for scoring roc auc score for multi-class
def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)

    for (idx, c_label) in enumerate(target):
        fpr, tpr, thresholds = roc_curve(y_test[:,idx].astype(int), y_pred[:,idx])
        c_ax.plot(fpr, tpr, label = '%s (AUC:%0.2f)'  % (c_label, auc(fpr, tpr)))
    c_ax.plot(fpr, fpr, 'b-', label = 'Random Guessing')
    z= roc_auc_score(y_test, y_test, average=average)
    print('ROC AUC score:', z)

    c_ax.legend()
    c_ax.set_xlabel('False Positive Rate')
    c_ax.set_ylabel('True Positive Rate')
    plt.show()
    import sklearn.metrics as skm

    TP, FN, FP, TN = skm.multilabel_confusion_matrix(y_test, y_pred)
    print('Outcome values : n', TP, FN, FP, TN)
    print(skm.classification_report(y_test, y_pred))

class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.layer1 = nn.Linear(latent_dims, 128)
        self.layer2 = nn.Linear(128, 80)
        self.layer3 = nn.Linear(80, 64)
        self.layer4 = nn.Linear(64, 16)
        self.layer_out = nn.Linear(16, 4)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        #inputs = inputs.view(-1, 1, IMGSIZE * IMGSIZE)
        x = self.relu(self.layer1(inputs))
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.relu(self.layer3(x))
        x = self.dropout(x)
        x = self.relu(self.layer4(x))
        x = self.sigmoid(self.layer_out(x))
        return x




def  DNN_train(DNN_model, trainloader, epochs=1):
    opt = torch.optim.Adam(DNN_model.parameters())
    for epoch in range(epochs):
        tot_loss = 0
        for x, y in trainloader:
            #y=F.one_hot(y, num_classes=4)
            y = torch.nn.functional.one_hot(y.to(torch.int64), 4)
            opt.zero_grad()
            y_hat = DNN_model(x)
            y = y.type(torch.FloatTensor)
            loss = criterion(y, y_hat)
            loss.backward()
            opt.step()
            tot_loss += loss.item()
        tot_loss = tot_loss / len(trainloader)
        print(' train: ',tot_loss)
    return DNN_model



def DNN_val(DNN_model, test_loader):
    preds = []
    labels = []
    with torch.no_grad():
        tot_loss = 0
        for x, y in test_loader:
            not_one_hot_y=y
            y = torch.nn.functional.one_hot(y.to(torch.int64), 4)
            y_hat = DNN_model(x)
            output = y_hat.max(1, keepdim=True)[1]
            loss = criterion(y_hat, y)
            tot_loss += loss.item()
            preds.append(output)
            labels.append(not_one_hot_y)
        tot_loss=tot_loss / len(test_loader)
        labels = labels[0].tolist()
        labels=[int(i) for i in labels]
        preds = preds[0].tolist()
        preds = [item for sublist in preds for item in sublist]
        accuracy = metrics.accuracy_score(labels, preds)
        print('ac :', accuracy)
        #auc = metrics.roc_auc_score(labels, preds,multi_class='ovo')
        #print('auc: ', auc)
        multiclass_roc_auc_score(labels,preds)
    return tot_loss

def use_encoder(data,autoencoder):
    encoded_imgs=[]
    images, labels = [item[0] for item in data], torch.FloatTensor([item[1] for item in data])
    for img,label in zip(images, labels):
        img_to_enc = torch.flatten(img)
        encoded_img=autoencoder.encoder(img_to_enc)
        #decoded_imgs.append(decoded_img)
        encoded_img= torch.FloatTensor(encoded_img.tolist())
        encoded_imgs.append(encoded_img)
    return encoded_imgs,labels


