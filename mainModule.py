# -*- coding: utf-8 -*-
"""
IRIS Classification
    - What is the Iris dataset? : http://en.wikipedia.org/wiki/Iris_flower_data_set
    - Author: Jaehyun Ahn (jaehyunahn@sogang.ac.kr)
    - Date: 2015. 05. 26
"""
from sklearn.cross_validation import cross_val_score
from sklearn.datasets import load_iris
from sklearn.preprocessing import scale
from nolearn.dbn import DBN
import numpy as np

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
print("Class 1", iris.data[1], iris.target[1])
# Standardize a dataset along any axis : Center to mean and component wise scale to unit variance.
scaled_data = scale(iris.data)
DeepBeliefNet.fit(scaled_data, iris.target)

# X is training sample, which is iris.data[1] element: [[original], [scaled variable]]
X = np.array([[4.9,  3. ,  1.4,  0.2], [-1.14301691e+00, -1.24957601e-01, -1.34127240e+00, -1.31297673e+00]])

#print(scaled_data)
print(DeepBeliefNet.predict(scaled_data))
print(DeepBeliefNet.predict(X))

scores = cross_val_score(DeepBeliefNet, scaled_data, iris.target, cv=10)
print ("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2))