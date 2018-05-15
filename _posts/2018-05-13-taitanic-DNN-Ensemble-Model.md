---
title: "kaggle: 타이타닉 생존자 예측 - DNN Ensemble Model"
header:
    overlay_image: /assets/images/2018-05-05-kaggle-taitanic/titanic_cover.jpg
    overlay_filter: 0.5
    caption: "Photo credit: [**Unsplash**](https://unsplash.com)"
categories:
    - kaggle
tags:
    - python
    - notebook
    - kaggle
    - titanic
toc: true
---

# 개요
- dnn 적용
- cross validation 으로 최적 hyper parameter 찾기
    - epoch, layers, units 등
- ensemble 적용
    - 최종 hyper parameter 기반으로 ensemble 구성

**In [1]:**

{% highlight python linenos %}
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import KFold
import time
import matplotlib.pyplot as plt
{% endhighlight %}

    /home/shkim/anaconda3/lib/python3.6/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
      from ._conv import register_converters as _register_converters

## model class

**In [2]:**

{% highlight python %}
class Model:

    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
    
    def build_net(self, features=8, layers=4, units=100, bn=False):
        with tf.variable_scope(self.name) as scope:
            # dropout (keep_prob) rate  0.7~0.5 on training, but should be 1
            # for testing
            self.training = tf.placeholder(tf.bool)
    
            # input place holders
            self.X = tf.placeholder(tf.float32, [None, features])
            self.Y = tf.placeholder(tf.float32, [None, 1])
            self.learning_rate = tf.placeholder(tf.float32)
            
            hiddenLayer = self.X
            for i in range(layers):
                if bn == True:
                    hiddenLayer = self.dense_batch_relu (hiddenLayer, units, 'layer'+str(i))
                else:
                    hiddenLayer = self.dense_relu_dropout (hiddenLayer, units, 'layer'+str(i))
                
            self.logits = tf.layers.dense(inputs=hiddenLayer, units=1)
        
        # define cost/loss & optimizer
        self.cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.Y))
        
        if bn == True:
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
        else:
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
    
        self.prediction = tf.round(tf.nn.sigmoid(self.logits))
        correct_prediction = tf.equal(self.prediction, self.Y)
        self.correct_count = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    def dense (self, x, size, scope):
        return tf.contrib.layers.fully_connected(x, size, activation_fn=None, scope=scope)
    
    def dense_batch_relu(self, x, size, scope):
        with tf.variable_scope(scope):
            h1 = tf.contrib.layers.fully_connected(x, size, activation_fn=None)
            h2 = tf.contrib.layers.batch_norm(h1, center=True, scale=True, is_training=self.training)
            return tf.nn.relu(h2)
        
    def dense_relu_dropout(self, x, size, scope):
        with tf.variable_scope(scope):
            initializer = tf.contrib.layers.xavier_initializer()
            hiddenLayer = tf.layers.dense(inputs=x, units=size, activation=tf.nn.relu, kernel_initializer=initializer)
            hiddenLayer = tf.layers.dropout(inputs=hiddenLayer, rate=0.5, training=self.training)
            return hiddenLayer
    
    def predict(self, x_test, training=False):
        return self.sess.run(self.prediction, feed_dict={self.X: x_test, self.training: training})
    
    def get_accuracy(self, x_test, y_test, training=False):
        return self.sess.run(self.accuracy, feed_dict={self.X: x_test, self.Y: y_test, self.training: training})
    
    def train(self, x_data, y_data, learning_rate, training=True):
        return self.sess.run([self.cost, self.optimizer], feed_dict={self.X: x_data, self.Y: y_data, self.learning_rate: learning_rate, self.training: training})
{% endhighlight %}

## cross validation 함수 

**In [3]:**

{% highlight python %}
def cross_valid (sess, train_x, train_y, foldCount, layers, units, epochs, learning_rate):
    
    fig = plt.figure(figsize=(18, 6))
    
    k_fold = KFold(foldCount)
    
    models_v = []
    for m in range(foldCount):
        models_v.append(Model(sess, "model_v" + str(m)))
        models_v[m].build_net(features=train_x.shape[1], layers=layers, units=units, bn=False)
    
    sess.run(tf.global_variables_initializer())
    
    start_time = time.time()
    
    valid_acc = []
    idx = 0;
    for train_indices, valid_indices in k_fold.split(train_x):
        #print('Train: %s \n| test: %s\n' % (train_indices, test_indices))
        train_acc_collect = []
        valid_acc_collect = []
    
        m = models_v[idx]
    
        start_time_k = time.time()
    
        X_train = train_x.iloc[train_indices].values
        Y_train = train_y.iloc[train_indices].values.reshape([-1,1])
        X_valid = train_x.iloc[valid_indices].values
        Y_valid = train_y.iloc[valid_indices].values.reshape([-1,1])
    
        fig.add_subplot(foldCount / 5, 5, idx+1)
        
        for i in range(epochs):
            c, _ = m.train(X_train, Y_train, learning_rate)
            accuracy_t = m.get_accuracy(X_train, Y_train)
            accuracy_v = m.get_accuracy(X_valid, Y_valid)
            train_acc_collect.append(accuracy_t)
            valid_acc_collect.append(accuracy_v)
    
        accuracy = m.get_accuracy(X_valid, Y_valid)
        print("[", idx, "] accuracy=%.2f" % accuracy, " %.2f seconds" % (time.time() - start_time_k))
        valid_acc.append(accuracy)
    
        plt.plot(train_acc_collect, "r")
        plt.plot(valid_acc_collect, "g")
        plt.ylim(0, 1)
    
        idx = idx + 1
    
    plt.show()
    print(" %.2f seconds" % (time.time() - start_time))
    return valid_acc, round(np.mean(valid_acc) * 100, 2)
{% endhighlight %}

## 학습데이터 읽어 오기
학습데이터는 아래의 코드로 전처리된 것을 사용한다.
: 들여쓰기
: [타이타닉 데이터분석](./김성헌_타이타닉_데이터분석.ipynb)

**In [4]:**
{% highlight python %}
train = pd.read_csv('train_preprocessed.csv')
train.head()
{% endhighlight %}


| Employee         | Salary |                                                              |
| --------         | ------ | ------------------------------------------------------------ |
| [John Doe](#)    | $1     | Because that's all Steve Jobs needed for a salary.           |
| [Jane Doe](#)    | $100K  | For all the blogging she does.                               |
| [Fred Bloggs](#) | $100M  | Pictures are worth a thousand words, right? So Jane × 1,000. |
| [Jane Bloggs](#) | $100B  | With hair like that?! Enough said.  



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Title</th>
      <th>FamilySize</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>0.8</td>
      <td>1</td>
      <td>2</td>
      <td>0.4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0</td>
      <td>1</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>0.8</td>
      <td>0</td>
      <td>2</td>
      <td>0.4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



**In [5]:**

{% highlight python %}
train_y = train['Survived']
train_x = train.drop('Survived', axis=1)
{% endhighlight %}

**In [6]:**

{% highlight python %}
train_y.head()
{% endhighlight %}




    0    0
    1    1
    2    1
    3    1
    4    0
    Name: Survived, dtype: int64



**In [7]:**

{% highlight python %}
type(train_y)
{% endhighlight %}




    pandas.core.series.Series



**In [8]:**

{% highlight python %}
train_x.head()
{% endhighlight %}




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Title</th>
      <th>FamilySize</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>0.8</td>
      <td>1</td>
      <td>2</td>
      <td>0.4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0</td>
      <td>1</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>0.8</td>
      <td>0</td>
      <td>2</td>
      <td>0.4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



## cross validation 으로 hyper parameter 결정 

**In [9]:**

{% highlight python %}
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

with tf.variable_scope("case1"):
    # hyper parameters
    learning_rate = 0.001
    training_epochs = 200
    layers = 1
    units = 100
    valid_acc, avg_acc = cross_valid (sess, train_x, train_y, 10, layers, units, training_epochs, learning_rate)
    print("avg_acc: ", avg_acc)
{% endhighlight %}

    WARNING:tensorflow:From /home/shkim/anaconda3/lib/python3.6/site-packages/tensorflow/contrib/learn/python/learn/datasets/base.py:198: retry (from tensorflow.contrib.learn.python.learn.datasets.base) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use the retry module or similar alternatives.
    [ 0 ] accuracy=0.79  0.74 seconds
    [ 1 ] accuracy=0.88  0.63 seconds
    [ 2 ] accuracy=0.76  0.69 seconds
    [ 3 ] accuracy=0.89  0.64 seconds
    [ 4 ] accuracy=0.81  0.66 seconds
    [ 5 ] accuracy=0.82  0.68 seconds
    [ 6 ] accuracy=0.79  0.67 seconds
    [ 7 ] accuracy=0.80  0.66 seconds
    [ 8 ] accuracy=0.89  0.65 seconds
    [ 9 ] accuracy=0.83  0.67 seconds



![png](/assets/images/2018-05-13-taitanic_2/2018-05-13-taitanic_2_13_1.png) 


     7.15 seconds
    avg_acc:  82.5


**In [10]:**

{% highlight python %}
with tf.variable_scope("case2"):
    # hyper parameters
    learning_rate = 0.001
    training_epochs = 200
    layers = 2
    units = 100
    valid_acc, avg_acc = cross_valid (sess, train_x, train_y, 10, layers, units, training_epochs, learning_rate)
    print("avg_acc: ", avg_acc)
{% endhighlight %}

    [ 0 ] accuracy=0.81  1.16 seconds
    [ 1 ] accuracy=0.89  1.13 seconds
    [ 2 ] accuracy=0.76  1.07 seconds
    [ 3 ] accuracy=0.88  1.08 seconds
    [ 4 ] accuracy=0.84  1.07 seconds
    [ 5 ] accuracy=0.84  1.11 seconds
    [ 6 ] accuracy=0.80  1.09 seconds
    [ 7 ] accuracy=0.81  1.13 seconds
    [ 8 ] accuracy=0.89  1.14 seconds
    [ 9 ] accuracy=0.85  1.05 seconds



![png](/assets/images/2018-05-13-taitanic_2/2018-05-13-taitanic_2_14_1.png) 


     11.48 seconds
    avg_acc:  83.73


**In [11]:**

{% highlight python %}
with tf.variable_scope("case3"):
    # hyper parameters
    learning_rate = 0.001
    training_epochs = 200
    layers = 3
    units = 100
    valid_acc, avg_acc = cross_valid (sess, train_x, train_y, 10, layers, units, training_epochs, learning_rate)
    print("avg_acc: ", avg_acc)
{% endhighlight %}

    [ 0 ] accuracy=0.82  1.95 seconds
    [ 1 ] accuracy=0.88  1.59 seconds
    [ 2 ] accuracy=0.76  1.59 seconds
    [ 3 ] accuracy=0.89  1.58 seconds
    [ 4 ] accuracy=0.83  1.58 seconds
    [ 5 ] accuracy=0.82  1.65 seconds
    [ 6 ] accuracy=0.80  1.65 seconds
    [ 7 ] accuracy=0.79  1.63 seconds
    [ 8 ] accuracy=0.89  2.26 seconds
    [ 9 ] accuracy=0.85  1.70 seconds



![png](/assets/images/2018-05-13-taitanic_2/2018-05-13-taitanic_2_15_1.png) 


     17.64 seconds
    avg_acc:  83.28


**In [12]:**

{% highlight python %}
with tf.variable_scope("case4"):
    # hyper parameters
    learning_rate = 0.001
    training_epochs = 200
    layers = 4
    units = 100
    valid_acc, avg_acc = cross_valid (sess, train_x, train_y, 10, layers, units, training_epochs, learning_rate)
    print("avg_acc: ", avg_acc)
{% endhighlight %}

    [ 0 ] accuracy=0.80  2.20 seconds
    [ 1 ] accuracy=0.87  2.07 seconds
    [ 2 ] accuracy=0.76  2.11 seconds
    [ 3 ] accuracy=0.89  2.09 seconds
    [ 4 ] accuracy=0.83  2.14 seconds
    [ 5 ] accuracy=0.85  2.20 seconds
    [ 6 ] accuracy=0.80  2.20 seconds
    [ 7 ] accuracy=0.79  2.15 seconds
    [ 8 ] accuracy=0.88  2.15 seconds
    [ 9 ] accuracy=0.85  2.14 seconds



![png](/assets/images/2018-05-13-taitanic_2/2018-05-13-taitanic_2_16_1.png) 


     21.88 seconds
    avg_acc:  83.17


**In [13]:**

{% highlight python %}
with tf.variable_scope("case5"):
    # hyper parameters
    learning_rate = 0.001
    training_epochs = 200
    layers = 5
    units = 100
    valid_acc, avg_acc = cross_valid (sess, train_x, train_y, 10, layers, units, training_epochs, learning_rate)
    print("avg_acc: ", avg_acc)
{% endhighlight %}

    [ 0 ] accuracy=0.81  2.61 seconds
    [ 1 ] accuracy=0.87  2.58 seconds
    [ 2 ] accuracy=0.78  2.56 seconds
    [ 3 ] accuracy=0.89  2.69 seconds
    [ 4 ] accuracy=0.83  2.77 seconds
    [ 5 ] accuracy=0.82  2.74 seconds
    [ 6 ] accuracy=0.80  2.93 seconds
    [ 7 ] accuracy=0.78  3.02 seconds
    [ 8 ] accuracy=0.88  2.77 seconds
    [ 9 ] accuracy=0.87  2.58 seconds



![png](/assets/images/2018-05-13-taitanic_2/2018-05-13-taitanic_2_17_1.png) 


     27.69 seconds
    avg_acc:  83.05


**In [14]:**

{% highlight python %}
with tf.variable_scope("case6"):
    # hyper parameters
    learning_rate = 0.001
    training_epochs = 200
    layers = 6
    units = 100
    valid_acc, avg_acc = cross_valid (sess, train_x, train_y, 10, layers, units, training_epochs, learning_rate)
    print("avg_acc: ", avg_acc)
{% endhighlight %}

    [ 0 ] accuracy=0.78  3.41 seconds
    [ 1 ] accuracy=0.87  3.45 seconds
    [ 2 ] accuracy=0.75  3.38 seconds
    [ 3 ] accuracy=0.83  3.39 seconds
    [ 4 ] accuracy=0.78  3.72 seconds
    [ 5 ] accuracy=0.83  3.32 seconds
    [ 6 ] accuracy=0.78  3.39 seconds
    [ 7 ] accuracy=0.81  3.29 seconds
    [ 8 ] accuracy=0.85  3.24 seconds
    [ 9 ] accuracy=0.85  3.32 seconds



![png](/assets/images/2018-05-13-taitanic_2/2018-05-13-taitanic_2_18_1.png) 


     34.35 seconds
    avg_acc:  81.26


**In [15]:**

{% highlight python %}
with tf.variable_scope("case7"):
    # hyper parameters
    learning_rate = 0.001
    training_epochs = 200
    layers = 7
    units = 100
    valid_acc, avg_acc = cross_valid (sess, train_x, train_y, 10, layers, units, training_epochs, learning_rate)
    print("avg_acc: ", avg_acc)
{% endhighlight %}

    [ 0 ] accuracy=0.71  3.74 seconds
    [ 1 ] accuracy=0.58  3.82 seconds
    [ 2 ] accuracy=0.78  3.73 seconds
    [ 3 ] accuracy=0.75  3.62 seconds
    [ 4 ] accuracy=0.78  3.71 seconds
    [ 5 ] accuracy=0.81  3.56 seconds
    [ 6 ] accuracy=0.78  3.66 seconds
    [ 7 ] accuracy=0.75  3.65 seconds
    [ 8 ] accuracy=0.81  3.72 seconds
    [ 9 ] accuracy=0.75  3.53 seconds



![png](/assets/images/2018-05-13-taitanic_2/2018-05-13-taitanic_2_19_1.png) 


     37.19 seconds
    avg_acc:  74.98


**In [16]:**

{% highlight python %}
with tf.variable_scope("case8"):
    # hyper parameters
    learning_rate = 0.001
    training_epochs = 200
    layers = 8
    units = 100
    valid_acc, avg_acc = cross_valid (sess, train_x, train_y, 10, layers, units, training_epochs, learning_rate)
    print("avg_acc: ", avg_acc)
{% endhighlight %}

    [ 0 ] accuracy=0.54  4.28 seconds
    [ 1 ] accuracy=0.58  4.19 seconds
    [ 2 ] accuracy=0.43  4.11 seconds
    [ 3 ] accuracy=0.54  4.12 seconds
    [ 4 ] accuracy=0.71  4.13 seconds
    [ 5 ] accuracy=0.71  4.24 seconds
    [ 6 ] accuracy=0.76  4.20 seconds
    [ 7 ] accuracy=0.39  4.10 seconds
    [ 8 ] accuracy=0.70  4.14 seconds
    [ 9 ] accuracy=0.67  4.29 seconds



![png](/assets/images/2018-05-13-taitanic_2/2018-05-13-taitanic_2_20_1.png) 


     42.25 seconds
    avg_acc:  60.39


**In [17]:**

{% highlight python %}
with tf.variable_scope("case9"):
    # hyper parameters
    learning_rate = 0.001
    training_epochs = 200
    layers = 9
    units = 100
    valid_acc, avg_acc = cross_valid (sess, train_x, train_y, 10, layers, units, training_epochs, learning_rate)
    print("avg_acc: ", avg_acc)
{% endhighlight %}

    [ 0 ] accuracy=0.48  4.74 seconds
    [ 1 ] accuracy=0.43  4.93 seconds
    [ 2 ] accuracy=0.42  4.68 seconds
    [ 3 ] accuracy=0.64  4.96 seconds
    [ 4 ] accuracy=0.60  4.67 seconds
    [ 5 ] accuracy=0.70  5.44 seconds
    [ 6 ] accuracy=0.75  4.93 seconds
    [ 7 ] accuracy=0.38  4.64 seconds
    [ 8 ] accuracy=0.34  5.26 seconds
    [ 9 ] accuracy=0.40  5.06 seconds



![png](/assets/images/2018-05-13-taitanic_2/2018-05-13-taitanic_2_21_1.png) 


     49.78 seconds
    avg_acc:  51.29


**In [18]:**

{% highlight python %}
with tf.variable_scope("case11"):
    # hyper parameters
    learning_rate = 0.001
    training_epochs = 200
    layers = 3
    units = 50
    valid_acc, avg_acc = cross_valid (sess, train_x, train_y, 10, layers, units, training_epochs, learning_rate)
    print("avg_acc: ", avg_acc)
{% endhighlight %}

    [ 0 ] accuracy=0.81  1.60 seconds
    [ 1 ] accuracy=0.89  1.65 seconds
    [ 2 ] accuracy=0.78  1.91 seconds
    [ 3 ] accuracy=0.89  1.64 seconds
    [ 4 ] accuracy=0.82  1.65 seconds
    [ 5 ] accuracy=0.83  1.71 seconds
    [ 6 ] accuracy=0.79  1.59 seconds
    [ 7 ] accuracy=0.81  1.66 seconds
    [ 8 ] accuracy=0.88  1.61 seconds
    [ 9 ] accuracy=0.84  1.60 seconds



![png](/assets/images/2018-05-13-taitanic_2/2018-05-13-taitanic_2_22_1.png) 


     17.09 seconds
    avg_acc:  83.28


**In [19]:**

{% highlight python %}
with tf.variable_scope("case12"):
    # hyper parameters
    learning_rate = 0.001
    training_epochs = 200
    layers = 3
    units = 150
    valid_acc, avg_acc = cross_valid (sess, train_x, train_y, 10, layers, units, training_epochs, learning_rate)
    print("avg_acc: ", avg_acc)
{% endhighlight %}

    [ 0 ] accuracy=0.80  2.64 seconds
    [ 1 ] accuracy=0.90  2.65 seconds
    [ 2 ] accuracy=0.75  2.67 seconds
    [ 3 ] accuracy=0.89  2.65 seconds
    [ 4 ] accuracy=0.84  2.66 seconds
    [ 5 ] accuracy=0.83  2.58 seconds
    [ 6 ] accuracy=0.80  2.62 seconds
    [ 7 ] accuracy=0.76  2.61 seconds
    [ 8 ] accuracy=0.89  2.63 seconds
    [ 9 ] accuracy=0.85  2.67 seconds



![png](/assets/images/2018-05-13-taitanic_2/2018-05-13-taitanic_2_23_1.png) 


     26.81 seconds
    avg_acc:  83.17


**In [20]:**

{% highlight python %}
with tf.variable_scope("case13"):
    # hyper parameters
    learning_rate = 0.001
    training_epochs = 200
    layers = 3
    units = 200
    valid_acc, avg_acc = cross_valid (sess, train_x, train_y, 10, layers, units, training_epochs, learning_rate)
    print("avg_acc: ", avg_acc)
{% endhighlight %}

    [ 0 ] accuracy=0.80  3.21 seconds
    [ 1 ] accuracy=0.89  3.19 seconds
    [ 2 ] accuracy=0.74  3.15 seconds
    [ 3 ] accuracy=0.84  3.16 seconds
    [ 4 ] accuracy=0.84  3.49 seconds
    [ 5 ] accuracy=0.84  3.31 seconds
    [ 6 ] accuracy=0.80  4.62 seconds
    [ 7 ] accuracy=0.79  3.46 seconds
    [ 8 ] accuracy=0.89  3.28 seconds
    [ 9 ] accuracy=0.85  3.54 seconds



![png](/assets/images/2018-05-13-taitanic_2/2018-05-13-taitanic_2_24_1.png) 


     34.88 seconds
    avg_acc:  82.83


**In [21]:**

{% highlight python %}
with tf.variable_scope("case14"):
    # hyper parameters
    learning_rate = 0.001
    training_epochs = 200
    layers = 3
    units = 250
    valid_acc, avg_acc = cross_valid (sess, train_x, train_y, 10, layers, units, training_epochs, learning_rate)
    print("avg_acc: ", avg_acc)
{% endhighlight %}

    [ 0 ] accuracy=0.80  3.98 seconds
    [ 1 ] accuracy=0.89  4.36 seconds
    [ 2 ] accuracy=0.75  4.11 seconds
    [ 3 ] accuracy=0.85  5.56 seconds
    [ 4 ] accuracy=0.84  4.18 seconds
    [ 5 ] accuracy=0.85  4.12 seconds
    [ 6 ] accuracy=0.80  4.11 seconds
    [ 7 ] accuracy=0.78  4.29 seconds
    [ 8 ] accuracy=0.89  4.05 seconds
    [ 9 ] accuracy=0.84  4.47 seconds



![png](/assets/images/2018-05-13-taitanic_2/2018-05-13-taitanic_2_25_1.png) 


     43.67 seconds
    avg_acc:  82.94


**In [22]:**

{% highlight python %}
with tf.variable_scope("case15"):
    # hyper parameters
    learning_rate = 0.001
    training_epochs = 200
    layers = 3
    units = 300
    valid_acc, avg_acc = cross_valid (sess, train_x, train_y, 10, layers, units, training_epochs, learning_rate)
    print("avg_acc: ", avg_acc)
{% endhighlight %}

    [ 0 ] accuracy=0.79  4.79 seconds
    [ 1 ] accuracy=0.89  4.90 seconds
    [ 2 ] accuracy=0.76  4.83 seconds
    [ 3 ] accuracy=0.85  5.48 seconds
    [ 4 ] accuracy=0.85  5.05 seconds
    [ 5 ] accuracy=0.85  5.07 seconds
    [ 6 ] accuracy=0.80  5.23 seconds
    [ 7 ] accuracy=0.80  5.30 seconds
    [ 8 ] accuracy=0.89  4.86 seconds
    [ 9 ] accuracy=0.85  4.85 seconds



![png](/assets/images/2018-05-13-taitanic_2/2018-05-13-taitanic_2_26_1.png) 


     50.81 seconds
    avg_acc:  83.39

 









## Ensemble Model 학습
hyper paramers
- learning_rate: 0.001
- training_epoches: 200
- layers: 3
- units:100

- ensemble models: 10 

**In [23]:**

{% highlight python %}
with tf.variable_scope("submission_model"):
    learning_rate = 0.001
    training_epochs = 200
    layers = 3
    units = 100

    ensembles = 10
    models = []
    for i in range(ensembles):
        model = Model(sess, "model" + str(i))
        model.build_net(features=train_x.shape[1], layers=layers, units=units, bn=False)
        models.append(model)
    
    sess.run(tf.global_variables_initializer())
    
    print('Learning Started!')
    start_time = time.time()
    
    cost_collect = np.zeros([len(models), training_epochs])
    
    for i in range(training_epochs):
        for m_idx, m in enumerate(models):
            c, _ = m.train(train_x.values, train_y.values.reshape([-1,1]), learning_rate)
            cost_collect[m_idx, i] = c;
            
    print('Learning Finished!')
    print("--- %.2f seconds ---" %(time.time() - start_time))
    
    fig = plt.figure(figsize=(18, 6))
    for m_idx in range(len(models)):
        fig.add_subplot(len(models) / 5, 5, m_idx+1)
        plt.plot(cost_collect[m_idx])
    plt.show() 
{% endhighlight %}

    Learning Started!
    Learning Finished!
    --- 17.66 seconds ---



![png](/assets/images/2018-05-13-taitanic_2/2018-05-13-taitanic_2_28_1.png) 


## 학습 데이터로 성능평가 

**In [24]:**

{% highlight python %}
predictions = np.zeros([train_x.shape[0],1])

for m_idx, m in enumerate(models):
    accuracy = m.get_accuracy(train_x.values, train_y.values.reshape([-1,1]))
    p = m.predict(train_x.values)
    print(m_idx, "accuracy: %.2f" % accuracy)
    predictions += p

predictions = predictions / len(models)
predictions_round = tf.round(predictions)
ensemble_correct = tf.equal(predictions_round, train_y.values.reshape([-1,1]))
ensemble_accuracy = tf.reduce_mean(tf.cast(ensemble_correct, tf.float32))
print('Ensemble accuracy: %.2f' % sess.run(ensemble_accuracy))
{% endhighlight %}

    0 accuracy: 0.85
    1 accuracy: 0.84
    2 accuracy: 0.85
    3 accuracy: 0.84
    4 accuracy: 0.85
    5 accuracy: 0.85
    6 accuracy: 0.85
    7 accuracy: 0.85
    8 accuracy: 0.85
    9 accuracy: 0.85
    Ensemble accuracy: 0.84

 









## 테스트 데이터로 prediction 

**In [25]:**

{% highlight python %}
test = pd.read_csv("test_preprocessed.csv")
test.head()
{% endhighlight %}




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Title</th>
      <th>FamilySize</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>892</td>
      <td>3</td>
      <td>0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>2</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>893</td>
      <td>3</td>
      <td>1</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0</td>
      <td>2</td>
      <td>0.4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>894</td>
      <td>2</td>
      <td>0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>2</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>895</td>
      <td>3</td>
      <td>0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>896</td>
      <td>3</td>
      <td>1</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0</td>
      <td>2</td>
      <td>0.8</td>
    </tr>
  </tbody>
</table>
</div>



**In [26]:**

{% highlight python %}
test_x = test.drop("PassengerId", axis=1)
test_x.head()
{% endhighlight %}




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Title</th>
      <th>FamilySize</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>2</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>1</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0</td>
      <td>2</td>
      <td>0.4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>2</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>1</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0</td>
      <td>2</td>
      <td>0.8</td>
    </tr>
  </tbody>
</table>
</div>



**In [27]:**

{% highlight python %}
test_predictions = np.zeros([test_x.shape[0],1])

for m_idx, m in enumerate(models):
    result = m.predict(test_x.values)
    test_predictions += result

test_predictions = test_predictions / len(models)
predictions_round = tf.round(test_predictions)
result = sess.run(predictions_round)
result = result.reshape([-1]).astype(int)

submission = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived":result})
submission.head()
{% endhighlight %}




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>892</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>893</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>894</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>895</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>896</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



**In [28]:**

{% highlight python %}
submission.to_csv('submission.csv', index=False)
{% endhighlight %}

## kaggle 결과!

![결과](/assets/images/2018-05-05-kaggle-taitanic/김성헌_타이타닉_DNN_Ensemble.png)

 
