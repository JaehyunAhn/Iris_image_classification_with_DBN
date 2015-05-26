## IRIS Classification with DBN(Deep Belief Net)
This project is for data classification with DBN Deep learning algorithm.

---
### Author Info
* Jaehyun Ahn (jaehyunahn@sogang.ac.kr)
* Creation Date: 15/05/26


### Version Info
* Python: 2.7
* Nolearn: 0.6 (Python 3.0 not supported)
* Numpy
* Sklearn


### Data Info
* What is IRIS Dataset? : Goto [Wiki](http://en.wikipedia.org/wiki/Iris_flower_data_set)

#### IRIS Flower Features
Iris flower data set has 3 classes, ecah of category has 50 samples with 4 features, which are sepal length/width and petal length/width.

---

### Training Info
 Using DBN, we create 4 hidden layer nodes, learning rate is 0.3, epochs executed 30 times.


#### Notes
 **Scaling** is important when you dealing iris data since width data were so subtle to differentiate. Thus if you want to use 'predict' function on your application, you have to know original data's scaling factors(mean / variance).