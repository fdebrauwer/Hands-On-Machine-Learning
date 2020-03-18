import numpy as np
from sklearn import datasets
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from keras.datasets import fashion_mnist
from keras.utils import np_utils
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import StratifiedKFold, cross_validate, learning_curve, ShuffleSplit
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, roc_auc_score


# Load & preprocessing
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
print("[INFO] loading Fashion MNIST...")
labels = ['T-shirt','Trousers','Pullover shirt','Dress','Coat','Sandals','Shirt','Sneaker','Bag','Ankle boot']

# reshape in 2D
im_shape = train_images.shape
train_images = train_images.reshape(60000,im_shape[1]*im_shape[2])
test_images = test_images.reshape(10000,im_shape[1]*im_shape[2])

# Standardization by centering and scaling of the images
scaler = StandardScaler()
train_images = scaler.fit_transform(train_images.astype('float32'))
test_images = scaler.fit_transform(test_images.astype('float32'))

# OneHotEncoding of the labels
# test_labels = np_utils.to_categorical(test_labels, 10)
# train_labels = np_utils.to_categorical(train_labels, 10)

# Data exploration
# for i in range(10):
#     plt.imshow(train_images[i])
#     plt.title(labels[train_labels[i]])
#     plt.axis('off')
#     plt.show()

model_set = [
    
    LogisticRegression(
        penalty='l2', 
        # specify the norm used in the penalizationThe ‘newton-cg’, ‘sag’ and ‘lbfgs’ solvers support only l2 penalties. 
        # ‘elasticnet’ is only supported by the ‘saga’ solver.
        dual=False, 
        # Prefer dual=False when n_samples > n_features.
        tol=0.0001, 
        # Tolerance for stopping criteria.
        C=1.0, 
        # Inverse of regularization strength; like in support vector machines, smaller values specify stronger regularization.
        class_weight=None,
        # The “balanced” mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data
        solver='newton-cg',
        # For small datasets, ‘liblinear’ is a good choice, whereas ‘sag’ and ‘saga’ are faster for large ones.
        # For multiclass problems, only ‘newton-cg’, ‘sag’, ‘saga’ and ‘lbfgs’ handle multinomial loss; 
        # ‘liblinear’ is limited to one-versus-rest schemes.
        # ‘newton-cg’, ‘lbfgs’, ‘sag’ and ‘saga’ handle L2 or no penalty
        # ‘liblinear’ and ‘saga’ also handle L1 penalty
        # ‘saga’ also supports ‘elasticnet’ penalty
        # ‘liblinear’ does not support setting penalty='none'
        max_iter=100,
        # max number of iterations
        multi_class='auto',
        # can take the values 'auto', 'ovr', 'multinomial'
        verbose=0,
        # how much print statement
        random_state=None, 
    )
    
    ,
    
    LinearSVC(
        penalty='l2', 
        loss='hinge', 
        dual=True, 
        tol=0.0001, 
        C=1.0, 
        multi_class='ovr', 
        fit_intercept=True, 
        intercept_scaling=1, 
        class_weight=None, 
        verbose=0, 
        random_state=None, 
        max_iter=100
    )
    
    ,
    
    SGDClassifier(
        loss='hinge', 
        # SVM losses‘hinge’, ‘log’, ‘modified_huber’, ‘squared_hinge’,‘perceptron’,
        # regression loss: ‘squared_loss’, ‘huber’, ‘epsilon_insensitive’, or ‘squared_epsilon_insensitive’
        penalty='l2',
        # Can use 'l1' or 'elasticnet' (linear combination of both) for more sparcity in the model, to make a feature selection
        alpha=0.00001, 
        # Constat that multiplies the regularization term
        max_iter=100,
        # maximum number of passes over the training data
        shuffle=True,
        # to shuffle the training data after each epoch, good to ensure a reduced variance and less overfit 
        # (during the training you see the true distribution of the data, no pattern in the sequence of batch seen)
        verbose=0,
        # to get some print statement during the training
        learning_rate='optimal',
        # Can be either 
        # 'constant' : eta = eta0
        # 'optimal' : eta = 1.0 / (alpha * (t + t0)) where t0 is chosen by a heuristic proposed by Leon Bottou.
        # 'invscaling' : eta = eta0 / pow(t, power_t)
        # 'adaptative': as long as the training keeps decreasing. Each time n_iter_no_change consecutive epochs 
        # fail to decrease the training loss by tol or fail to increase validation score by tol if early_stopping 
        # is True, the current learning rate is divided by 5.
        early_stopping=True,
        # To stop the algorithm quicker when the validation score is not improving for at least n_iter_no_change
        validation_fraction = 0.1,
        # The fraction of the training data that serve for the validation
        n_iter_no_change=5
        # number of iterations with no improvement to wait before early stopping
    ) 
    
    , 
    
    SVC(
        C=1.0, 
        # regulariztaion parameter
        kernel='rbf', 
        # Specifies the kernel type to be used in the algorithm. 
        # It must be one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ or a callable
        degree=3, 
        # Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels.
        gamma='scale', 
        # Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.
        # if gamma='scale' (default) is passed then it uses 1 / (n_features * X.var()) as value of gamma,
        # if ‘auto’, uses 1 / n_features.
        coef0=0.0, 
        # Independent term in kernel function. It is only significant in ‘poly’ and ‘sigmoid’.
        shrinking=True, 
        # Whether to use the shrinking heuristic.
        probability=False, 
        # Whether to enable probability estimates. This must be enabled prior to calling fit, 
        # will slow down that method as it internally uses 5-fold cross-validation, 
        # and predict_proba may be inconsistent with predict. 
        tol=0.001, 
        # Tolerance for stopping criterion.
        cache_size=200, 
        # Specify the size of the kernel cache (in MB).
        class_weight=None, 
        # The “balanced” mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data as n_samples / (n_classes * np.bincount(y))
        verbose=False, 
        # verbose
        max_iter=-1, 
        # Hard limit on iterations within solver, or -1 for no limit.
        decision_function_shape='ovr', 
        # Whether to return a one-vs-rest (‘ovr’) decision function of shape (n_samples, n_classes) as 
        # all other classifiers, or the original one-vs-one (‘ovo’) decision function of libsvm which has shape 
        # (n_samples, n_classes * (n_classes - 1) / 2). However, one-vs-one (‘ovo’) is always used as multi-class strategy.
        break_ties=False, 
        # If true, decision_function_shape='ovr', and number of classes > 2, predict will break ties according to 
        # he confidence values of decision_function; otherwise the first class among the tied classes is returned. 
        # Please note that breaking ties comes at a relatively high computational cost compared to a simple predict.
        random_state=None
    )
]

scoring = {
    'accuracy' : make_scorer(accuracy_score),
    'precision' : make_scorer(precision_score, average='weighted'),
    'recall' : make_scorer(recall_score, average='weighted'),
    'f1' : make_scorer(f1_score, average='weighted'),
}

max_value = 0

for model in model_set:
    
    print('[INFO] loading :', model)
        
    scores = cross_validate(
        estimator=model,
        X=train_images,
        y=train_labels,
        cv=3,
        scoring=scoring,
        verbose=1,
        return_estimator=False
    )    
    
    for key in scores.keys():
        print(key, ':', round(np.average(scores[key]), 4)*100, '%')
        
    # TODO build the learning graph for the best model
    if scores['f1']>max_value:
        max_value=scores['f1']
        best_model = model
        
print(best_model)

model.fit(train_images, train_labels)
pred = model.predict(test_images)

def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    axes : array of 3 axes, optional (default=None)
        Axes to use for plotting the curves.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt

fig, axes = plt.subplots(3, 2, figsize=(10, 15))
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
estimator = best_model
plot_learning_curve(
    estimator, 
    title="Learning Curves", 
    X=train_images, 
    y=train_labels, 
    axes=axes[:, 0], 
    ylim=(0.7, 1.01),
    cv=cv, 
    n_jobs=4
)

plt.show()
    
    # TODO run a grid search to determine the best hyperparameter value
    # what about the p∏recision recall tradoff in multiclass problems ? We cant move a treshold when the classes are mutually exclusive...