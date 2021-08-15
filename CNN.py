import math
import os
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.preprocessing import LabelBinarizer

criterion = nn.CrossEntropyLoss()

target = ['glioma', 'meningioma', 'no-tumor', 'pituitary']

fig, c_ax = plt.subplots(1, 1, figsize=(12, 8))

#this func accepts predictions and 'true' labels and creates AUC-ROC plot+ conf matrix
def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)

    for (idx, c_label) in enumerate(target):
        fpr, tpr, thresholds = roc_curve(y_test[:, idx].astype(int), y_pred[:, idx])
        c_ax.plot(fpr, tpr, label='%s (AUC:%0.2f)' % (c_label, auc(fpr, tpr)))
    c_ax.plot(fpr, fpr, 'b-', label='Random Guessing')
    z = roc_auc_score(y_test, y_test, average=average)
    print('ROC AUC score:', z)

    c_ax.legend()
    c_ax.set_xlabel('False Positive Rate')
    c_ax.set_ylabel('True Positive Rate')
    plt.show()
    import sklearn.metrics as skm

    TP, FN, FP, TN = skm.multilabel_confusion_matrix(y_test, y_pred)
    print('Outcome values : n', TP, FN, FP, TN)
    print(skm.classification_report(y_test, y_pred))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(8192, 4000)
        self.fc2 = nn.Linear(4000, 1000)
        self.fc3 = nn.Linear(1000, 300)
        self.fc4 = nn.Linear(300, 50)
        self.fc5 = nn.Linear(50, 4)

        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2,))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2,))
        x = F.relu(self.conv4(x))
        x = x.view(-1, x.size()[1] * x.size()[2] * x.size()[3])
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        output = F.sigmoid(self.fc5(x))
        return output

#this function is the training of the model
def train(model, train_loader, epochs=20):
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    losses = []
    #loop through data as num of epochs
    for epoch in range(epochs):
        tot_loss = 0
        for batch_idx, (data_x, labels) in enumerate(train_loader):
            opt.zero_grad()
            #predict y
            output = model(data_x)
            #calculate the loss
            loss = criterion(output, labels)
            #bp
            loss.backward()
            opt.step()
            tot_loss += loss.item()
            losses.append(loss)
        #calculates total loss
        tot_loss = tot_loss / len(train_loader)
    return model


#this is the test of the model
def test(model, test_loader):
    tot_loss = 0
    preds = []
    labels = []
    with torch.no_grad():
        for batch_idx, (data_x, label) in enumerate(test_loader):
            #use the trained model
            output = model(data_x)
            output = output.type(torch.FloatTensor)
            #calculte loss
            loss = criterion(output, label)
            tot_loss += loss.item()
            #pick the highest prob for label
            output = output.max(1, keepdim=True)[1]
            preds.append(output)
            labels.append(label)
        tot_loss = tot_loss / len(test_loader)
    labels = labels[0].tolist()
    preds = preds[0].tolist()
    preds = [item for sublist in preds for item in sublist]
    #calculate accuracy
    accuracy = metrics.accuracy_score(labels, preds)
    print('accuracy: ', accuracy)
    # plot AUC-ROC curve
    multiclass_roc_auc_score(labels, preds)


if __name__ == '__main__':

    transform_d = torchvision.transforms.Compose([torchvision.transforms.CenterCrop(64),
                                                  torchvision.transforms.Grayscale(1),
                                                  torchvision.transforms.ToTensor()])
    train_path = os.getcwd() + "\\images2"

    # import data and insert to tensors
    imagenet_data = torchvision.datasets.ImageFolder(train_path, transform=transform_d)
    #divide data to train and test
    split_len = [int(math.floor(len(imagenet_data) * 0.9)), int(math.ceil(len(imagenet_data) * 0.1))]
    trainData, valData = torch.utils.data.random_split(imagenet_data, split_len)

    #create data loaders
    train_loader = torch.utils.data.DataLoader(trainData, batch_size=30, shuffle=True)
    val_loader = torch.utils.data.DataLoader(valData, batch_size=30, shuffle=True)

    model = Net()   #create network object
    train_model = train(model, train_loader)
    preds = test(train_model, val_loader)
