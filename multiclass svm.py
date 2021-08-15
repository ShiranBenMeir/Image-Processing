import random
import numpy
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, roc_curve, auc, roc_auc_score
from sklearn.model_selection import train_test_split
from PIL import Image
import glob

from sklearn.preprocessing import LabelBinarizer
from sklearn.svm import SVC
from sklearn import svm, datasets
import sklearn.model_selection as model_selection
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

target= ['glioma','meningioma','no-tumor','pituitary']


fig, c_ax = plt.subplots(1,1, figsize = (12, 8))

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


# 1-glioma 2-meningioma 3-pituitary 0-no_tumor

imgs = []
labels = []
for filename in glob.glob('images2/glioma/*.jpg'):
    im = Image.open(filename)
    im = im.resize((174, 230))
    imgs.append([im, "glioma"])
for filename in glob.glob('images2/meningioma/*.jpg'):
    im = Image.open(filename)
    im = im.resize((174, 230))
    imgs.append([im, "meningioma"])
for filename in glob.glob('images2/pituitary/*.jpg'):
    im = Image.open(filename)
    im = im.resize((174, 230))
    imgs.append([im, "pituitary"])
for filename in glob.glob('images2/no_tumor/*.jpg'):
    im = Image.open(filename)
    im = im.resize((174, 230))
    imgs.append([im, "no_tumor"])
random.shuffle(imgs)
trainImgs, testImgs = train_test_split(imgs, test_size=0.3)

x_train = []
y_train = []
for example in trainImgs:
    ex = numpy.array(example[0])
    ex = ex.reshape(-1)
    x_train.append(ex)
    y_train.append(example[1])

x_test = []
y_test = []
for example in testImgs:
    ex = numpy.array(example[0])
    ex = ex.reshape(-1)
    x_test.append(ex)
    y_test.append(example[1])
print("hi")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from keras.datasets import mnist
from sklearn.datasets import load_iris
from numpy import reshape
import seaborn as sns
import pandas as pd

'''''
tsne = TSNE(n_components=3, verbose=1, random_state=123)
z = tsne.fit_transform(x_train)

df = pd.DataFrame()
df["y"] = y_train
df["comp-1"] = z[:,0]
df["comp-2"] = z[:,1]
df["comp-3"] = z[:,2]

df = pd.DataFrame()
df["y"] = y_train
df["comp-1"] = z[:, 0]
df["comp-2"] = z[:, 1]
df["comp-3"] = z[:, 2]

sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                palette=sns.color_palette("hls", 4),
                data=df).set(title="Brain tumor t-sne")
plt.show()

sns.scatterplot(x="comp-1", y="comp-3", hue=df.y.tolist(),
                palette=sns.color_palette("hls", 4),
                data=df).set(title="Brain tumor t-sne")
plt.show()

sns.scatterplot(x="comp-2", y="comp-3", hue=df.y.tolist(),
                palette=sns.color_palette("hls", 4),
                data=df).set(title="Brain tumor t-sne")
plt.show()


'''''
rbf = svm.SVC(kernel='rbf', gamma=0.5, C=0.1).fit(x_train, y_train)
poly = svm.SVC(kernel='poly', degree=3, C=1).fit(x_train, y_train)

poly_pred = poly.predict(x_test)
rbf_pred = rbf.predict(x_test)

poly_accuracy = accuracy_score(y_test, poly_pred)
poly_f1 = f1_score(y_test, poly_pred, average='weighted')
print('Accuracy : ', "%.2f" % (poly_accuracy*100))
print('F1 (Polynomial Kernel): ', "%.2f" % (poly_f1*100))
multiclass_roc_auc_score(y_test, poly_pred)


'''''
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(4):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], poly_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot of a ROC curve for a specific class
for i in range(4):
    plt.figure()
    plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

'''''