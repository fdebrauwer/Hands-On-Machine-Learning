import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from keras.datasets import fashion_mnist
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, roc_auc_score


# Load & preprocessing
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
labels = ['Dress','Trousers','Pull','Dress2','Coat','Heels','Sweater','Shoes','Other','Boot']

im_shape = train_images.shape
train_images = train_images.reshape(60000,im_shape[1]*im_shape[2])
test_images = test_images.reshape(10000,im_shape[1]*im_shape[2])

scaler = StandardScaler()
train_images = scaler.fit_transform(train_images.astype('float32'))
test_images = scaler.fit_transform(test_images.astype('float32'))

# Data exploration
# for i in range(10):
#     plt.imshow(train_images[i])
#     plt.title(labels[train_labels[i]])
#     plt.axis('off')
#     plt.show()

# print('Train images shape:', train_images.shape)
# print('Test images shape:', test_images.shape)
# print('Train labels shape:', train_labels.shape)
# print('Test label shape:', train_labels.shape)

model_set = [
    SGDClassifier(max_iter=100),
    RandomForestClassifier()
]
model_index = 0

skf = StratifiedKFold(n_splits=3, random_state=42, shuffle=True)

for model in model_set:
    matrix = []
    accuracy = []
    precision = []
    recall = []
    f1 = []
    roc = []
    
    for train_index, test_index in skf.split(train_images, train_labels):        
        x_train_fold = train_images[train_index]
        y_train_fold = train_labels[train_index]
        x_test_fold = train_images[test_index]
        y_test_fold = train_labels[test_index]
        
        # clone_clf.fit(x_train_fold, y_train_fold) #whats the benefit of cloning ?
        
        model.fit(x_train_fold, y_train_fold)
        y_pred_fold = model.predict(x_test_fold)
        
        matrix.append(confusion_matrix(y_test_fold, y_pred_fold))
        accuracy.append(accuracy_score(y_test_fold, y_pred_fold))
        precision.append(precision_score(y_test_fold, y_pred_fold, average='weighted')) # what are the different average method and their effect on the score
        recall.append(recall_score(y_test_fold, y_pred_fold, average='weighted'))
        f1.append(f1_score(y_test_fold, y_pred_fold, average='weighted'))
        # roc.append(roc_auc_score(y_test_fold, y_pred_fold, average='weighted'))
        
    print(model_set[model_index])
    model_index+=1
    print('Accuracy:', round(np.average(accuracy), 2)*100, '%')
    print('Precision:', round(np.average(precision), 2)*100, '%')
    print('Recall:', round(np.average(recall), 2)*100, '%')
    # print('ROC score:', round(np.average(roc), 2)*100, '%')
    print('Matrix:', matrix[0]) #average the different matrix
    plt.matshow(matrix[0])
    plt.show()
    row_sum = matrix[0].sum(axis=1, keepdims=True)
    matrix_absolute = matrix[0]/row_sum
    plt.matshow(matrix_absolute)
    plt.show()

# what are the different scoring classifiers vs non scoring classifier
# what about the recision recall tradoff in multiclass problems ? We cant move a treshold when the classes are mutually exclusive...