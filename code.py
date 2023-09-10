import matplotlib.pyplot as plt

from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import utils

digits = datasets.load_digits()

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)
plt.show()

n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

clf = svm.SVC(gamma=0.001)

#X_train, X_test, y_train, y_test = train_test_split(
#    data, digits.target, test_size=0.5, shuffle=False
#)
#clf.fit(X_train, y_train)
#predicted = clf.predict(X_test)

def train_dev_test_split(X, Y, test_size, dev_size):
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size = test_size + dev_size, shuffle = False
    )
    dev_train, X_test, dev_test, Y_test = train_test_split(
        X_test, Y_test, test_size= dev_size/(dev_size + test_size), shuffle = False
    )
    return X_train, X_test, Y_train, Y_test, dev_train, dev_test

def predict_and_eval(model, X_test, Y_test):
    model.fit(X_train, Y_train)
    predicted_val = model.predict(X_test)
    return predicted_val

X_train, X_test, Y_train, Y_test, dev_train, dev_test = train_dev_test_split(data, digits.target, 0.3, 0.3)


_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
predictions = predict_and_eval(clf, X_test, Y_test)
for ax, image, prediction in zip(axes, X_test, predictions):
    ax.set_axis_off()
    image = image.reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"Prediction: {prediction}")
plt.show()

print(
    f"Classification report for classifier {clf}:\n"
    f"{metrics.classification_report(Y_test, predictions)}\n"
)

y_true = []
y_pred = []
cm = confusion_matrix(Y_test, predictions)

for gt in range(len(cm)):
    for pred in range(len(cm)):
        y_true += [gt] * cm[gt][pred]
        y_pred += [pred] * cm[gt][pred]

print(
    "Classification report rebuilt from confusion matrix:\n"
    f"{metrics.classification_report(y_true, y_pred)}\n"
)

X = data
y  = digits.target
gamma = [0.001,0.01,0.1,1,10,100]
C = [0.1,1,2,5,10]
parameter_combinations = [{"gamma":i, "C":j} for i in gamma for j in C] 

test_sizes = [0.1, 0.2, 0.3] 
dev_sizes  = [0.1, 0.2, 0.3]
test_dev_size_groups = [{"test_size":i, "dev_size":j} for i in test_sizes for j in dev_sizes] 

clf = svm.SVC

result = utils.tune_hparams(clf,X_train, X_test, dev_train , Y_train, Y_test, dev_test, parameter_combinations)

for test_dev_size in test_dev_size_groups:
    X_train, X_test, dev_train , Y_train, Y_test, dev_test = train_dev_test_split(X,y,**test_dev_size)
    train_acc, dev_acc, test_acc, optimal_param = result
    _ = 1 - (sum(test_dev_size.values()))
    print(f'test_size: {test_dev_size["test_size"]}, dev_size: {test_dev_size["dev_size"]}, train_size: {_}, train_acc: {train_acc}, dev_acc: {dev_acc}, test_acc: {test_acc}, optimal_param: {optimal_param}')