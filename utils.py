from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import datasets, metrics, svm

def find_accuracy(model, X_test, Y_test):
    predicted = model.predict(X_test)
    return metrics.accuracy_score(Y_test,predicted)

def tune_hparams(model,X_train, X_test, dev_train , Y_train, Y_test, dev_test, list_of_param_combination):
    best_accuracy = -1
    for param_group in list_of_param_combination:
        temp_model = model(**param_group)
        temp_model.fit(X_train,Y_train)
        accuracy = find_accuracy(temp_model,dev_train,dev_test) 
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = temp_model
            optimal_param = param_group
    train_accuracy= find_accuracy(best_model,X_train,Y_train) 
    dev_accuracy = find_accuracy(best_model,dev_train,dev_test) 
    test_accuracy =  find_accuracy(best_model,X_test,Y_test)
    return train_accuracy, dev_accuracy, test_accuracy, optimal_param