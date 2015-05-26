# -*- coding: utf-8 -*-
from sklearn.cross_validation import cross_val_score
from sklearn.datasets import load_iris
from sklearn.preprocessing import scale
from nolearn.dbn import DBN
import numpy as np

__author__ = 'Jaehyun Ahn'

"""
    Load and return the iris dataset (classification).

    The iris dataset is a classic and very easy multi-class classification
    dataset.

    =================   ==============
    Classes                          3
    Samples per class               50
    Samples total                  150
    Dimensionality                   4
    Features            real, positive
    =================   ==============

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification labels,
        'target_names', the meaning of the labels, 'feature_names', the
        meaning of the features, and 'DESCR', the
        full description of the dataset.

    Examples
    --------
    Let's say you are interested in the samples 10, 25, and 50, and want to
    know their class name.

    >>> from sklearn.datasets import load_iris
    >>> data = load_iris()
    >>> data.target[[10, 25, 50]]
    array([0, 0, 1])
    >>> list(data.target_names)
    ['setosa', 'versicolor', 'virginica']
"""
iris = load_iris()
print ("l. setosa's sepal length / sepal width / petal length / petal width")
print (list(iris.data[2]))

"""
 DBN( layer_sizes, learning_rates, epochs)
    - layer_sizes:      n_vis_units, n_hid_units, ... , n_out_units
    - learning_rates:   entry per weight layer
    - epochs: number of epochs to train with back propagation
"""
DeepBeliefNet = DBN(
    [4, 4, 3],
    learn_rates=0.3,
    epochs=30,
)
print(iris.data[51], iris.target[51])
scaled_data = scale(iris.data)
DeepBeliefNet.fit(scaled_data, iris.target)
# [Error] Broadcast input array : The error message we look at in this section tells us that we
#                                   violated the broadcasting rules.
print(DeepBeliefNet.predict(scaled_data))

scores = cross_val_score(DeepBeliefNet, scaled_data, iris.target, cv=10)
print ("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2))