---
title: SVMs Sample Code
commentable: flase
Edit: 2021-12-15
mathjax: true
mermaid: true
tags: SVM Kernel
categories: Data
description: This is a post for sample code [SVMs](https://github.com/jyang-zhou/SVMs).
---

# SVMs
SVMs via sub-gradient descent and quadratic programming with sentiment analysis on tweets on US airline service quality. We focus on using Linear SVM, Kernel SVM with linear kernel and Kernel SVM with Gaussian kernel on the dataset.

Please go to [SVMs](https://github.com/jyang-zhou/SVMs) for the complete code.

## Package Needed
```
import numpy as np
import numpy.random as npr

import matplotlib.cm as cm
import matplotlib.pyplot as plt
%matplotlib inline

from math import sqrt

import csv
import cvxopt

def create_download_file(fname, preds):
    """Create file with predictions written as a csv file
    """
    ofile  = open(fname, "w")  
    writer = csv.writer(
        ofile, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL
    )
    writer.writerow(['id', 'category'])
    for i in range(preds.shape[0]):
        writer.writerow([i, preds[i]])


def plot_decision_countour(svm, X, y, grid_size=100):
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_size),
                         np.linspace(y_min, y_max, grid_size),
                         indexing='ij')
    data = np.stack([xx, yy], axis=2).reshape(-1, 2)
    pred = svm.predict(data).reshape(xx.shape)
    plt.contourf(xx, yy, pred,
                 cmap=cm.Paired,
                 levels=[-0.001, 0.001],
                 extend='both',
                 alpha=0.8)
    flatten = lambda m: np.array(m).reshape(-1,)
    plt.scatter(flatten(X[:,0][y==-1]),flatten(X[:,1][y==-1]),
                  c=flatten(y)[y==-1],cmap=cm.Paired,marker='o')
    plt.scatter(flatten(X[:,0][y==1]),flatten(X[:,1][y==1]),
                  c=flatten(y)[y==1],cmap=cm.Paired,marker='+')
    
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.plot()




def test_SVM(svm, num_samples=500,linear=False):
    """test svm
    """
    np.random.seed(783923)

    X = npr.random((num_samples, 2)) * 2 - 1
    if linear:
      y = 2 * (X.sum(axis=1) > 0) - 1.0
    else: 
      y = 2 * ((X ** 2).sum(axis=1) - 0.5 > 0) - 1.0
    svm.fit(X,y)
    
    plot_decision_countour(svm, X, y)

    from datetime import datetime
    np.random.seed(int(round(datetime.now().timestamp())))

def compute_acc(model, X, y):
    pred = model.predict(X)
    size = len(y)
    num_correct = (pred == y).sum()
    acc = num_correct / size
    print("{} out of {} correct, acc {:.3f}".format(num_correct, size, acc))
```

## Linear SVM
```
class LinearSVM():
    def __init__(self,C):
        """initialize the svm
        
        Args:
            C: weight associated with the hinge loss term in the SVM loss
        """
        self.w = None
        self.bias = None
        self.C = C

    def fit(self, X, y,num_epochs=30,lr_sched=lambda t: 0.1/t):
        """Fit the model on the data
        
        Args:
            X: [N x d] data matrix
            y: [N, ] array of labels
            num_epochs: number of passes over the training data we make
            lr_sched: function determining how the learning rate decays across
                      epochs
        
        Returns:
            self, in case you want to build a pipeline
        """
        assert np.ndim(X) == 2, 'data matrix X expected to be 2d'
        assert np.ndim(y) == 1, 'labels expected to be 1d'
        N, d = X.shape
        assert N == y.shape[0], 'expect [N, d] data matrix and [N] labels'
        self.w = np.zeros([d,1])
        self.bias = 0
        # TODO: implement a subgradient descent
        y = y.reshape((N,1))
        for n in range(N//num_epochs):
          h = y*(np.dot(X,self.w)+self.bias)
          a = np.random.randint(N - num_epochs)
          for i in range(a,a+num_epochs):
            if (h[i] < 1):
              xy = (y[i]*X[i])
              xy = xy.reshape(d,1)
              self.w = self.w + lr_sched(num_epochs) * xy
              self.bias = self.bias + lr_sched(num_epochs) * y[i]
            else:
              self.w = self.w
              self.bias = self.bias
        
        print("training complete")
        return self

    def predict(self, X, binarize=True):
        """make a prediction and return either the confidence margin or label
        
        Args:
            X: [N, d] array of data or [d,] single data point
            binarize: if True, then return the label, else the confidence margin
        
        Returns:
            Either confidence margin or predicted label
        """
        if self.w is None:
            raise ValueError("go fit the data first")
        X = np.atleast_2d(X)
        assert X.shape[1] == self.w.shape[0]
        res = X.dot(self.w)
        res = res.squeeze()+self.bias
        if binarize:
            return (res > 0).astype(np.int32) * 2 - 1
        else:
            return res

    def clone(self):
        """construct a fresh copy of myself
        """
        return LinearSVM(self.lamb, self.num_epochs)
```

## linear kernel and Gaussian Kernels
```
class Kernel(object):
    """
    A class containing all kinds of kernels.
    Note: the kernel should work for both input (Matrix, vector) and (vector, vector)
    """
    @staticmethod
    def linear():
        def f(x, y):
            return np.dot(x, y)
        return f

    @staticmethod
    def gaussian(sigma):
        def f(x, y):
            exponent = - (1/sigma**2) * np.linalg.norm((x-np.transpose(y)).transpose(), 2, 0) ** 2
            return np.exp(exponent)
        return f

    @staticmethod
    def _poly(dimension, offset):
        def f(x, y):
            return (offset + np.dot(x, y)) ** dimension
        return f

    @staticmethod
    def inhomogenous_polynomial(dimension):
        return Kernel._poly(dimension=dimension, offset=1.0)

    @staticmethod
    def homogenous_polynomial(dimension):
        return Kernel._poly(dimension=dimension, offset=0.0)

    @staticmethod
    def hyperbolic_tangent(kappa, c):
        def f(x, y):
            return np.tanh(kappa * np.dot(x, y) + c)
        return f
```

## Kernel SVM
```
class KernelSVM(object):
    def __init__(self, kernel, C):
        """
        Build a SVM given kernel function and C

        Parameters
        ----------
        kernel : function
            a function takes input (Matrix, vector) or (vector, vector)
        C : a scalar
            balance term

        Returns
        -------
        """
        self._kernel = kernel
        self.C = C

    def fit(self, X, y):
        """
        Fit the model given data X and ground truth label y

        Parameters
        ----------
        X : 2D array
            N x d data matrix (row per example)
        y : 1D array
            class label

        Returns
        -------
        """
        # Solve the QP problem to get the multipliers
        lagrange_multipliers = self._compute_multipliers(X, y)
        # Get all the support vectors, support weights and bias
        self._construct_predictor(X, y, lagrange_multipliers)
    
    def predict(self, X):
        """
        Predict the label given data X

        Parameters
        ----------
        X : 2D array
            N x d data matrix (row per example)

        Returns
        -------
        y : 1D array
            predicted label
        """
        result = np.full(X.shape[0], self._bias) # note: intializing scores with b
        for z_i, x_i, y_i in zip(self._weights,
                                 self._support_vectors,
                                 self._support_vector_labels):
            result += z_i * y_i * self._kernel(X, x_i) # the result is \sum_i alpha_i*y_i*x_i+b
        return np.sign(result)

    def _kernel_matrix(self, X):
        """
        Get the kernel matrix.

        Parameters
        ----------
        X : 2D array
            N x d data matrix (row per example)

        Returns
        -------
        K : 2D array
            N x N kernel matrix, where K[i][j] = inner_product(phi(i), phi(j))
        """
        K = self._kernel(X,np.transpose(X))
        return K
        pass

    def _construct_predictor(self, X, y, lagrange_multipliers):
        """
        Given the data, label and the multipliers, extract the support vectors and calculate the bias

        Parameters
        ----------
        X : 2D array
            N x d data matrix (row per example)
        y : 1D array
            class label
        lagrange_multipliers: 1D array
            the solution of lagrange_multiplier

        Returns
        -------
        """
        support_vector_indices = \
            lagrange_multipliers > 1e-5
            
        print("SV number: ", np.sum(support_vector_indices))

        support_multipliers = lagrange_multipliers[support_vector_indices]
        support_vectors = X[support_vector_indices]
        support_vector_labels = y[support_vector_indices]

        """
        Get the bias term
        """
        # TODO: implement
        #N,d = support_vectors.shape
        #bias = 0
        #for i in range(N):
        #  K = self._kernel(X, X[1])
        #  for j in range(N):
        #    bias = bias + support_multipliers[j] * support_vector_labels[j] * K[j]
        #bias = (np.sum(y) - bias)/N
        K = self._kernel_matrix(support_vectors)
        weights = support_multipliers * support_vector_labels
        y_hat = np.dot(weights,K)
        npsum = np.sum(support_vector_indices)
        bias = 1/npsum * np.sum((support_vector_labels - y_hat))
        self._bias=bias
        self._weights=support_multipliers
        self._support_vectors=support_vectors
        self._support_vector_labels=support_vector_labels


    def _compute_multipliers(self, X, y):
        """
        Given the data, label, solve the QP program to get lagrange multiplier.

        Parameters
        ----------
        X : 2D array
            N x d data matrix (row per example)
        y : 1D array
            class label

        Returns
        lagrange_multipliers: 1D array
        -------
        """
        N, d = X.shape

        K = self._kernel_matrix(X)
        """
        The standard QP solver formulation:
        min 1/2 alpha^T H alpha + f^T alpha
        s.t.
        A * alpha \coneleq a (A is former G)
        B * alpha = b
        """
        # TODO: implement. Specifically, define the H, f, A, a, B, b arguments
        # as indicated above.
        b = cvxopt.matrix(0.0)
        B = cvxopt.matrix(y,(1,N))
       
        A = cvxopt.matrix((np.diag(np.ones(N))))
        a = cvxopt.matrix((np.ones((N,1))*self.C))
        
        f = cvxopt.matrix(-1*np.ones([N]))
        H = cvxopt.matrix(np.outer(y,y)*K)

        solution = cvxopt.solvers.qp(H, f, A, a, B, b)

        # Lagrange multipliers
        return np.ravel(solution['x'])
```

# Data Set

- The [data](https://github.com/jyang-zhou/SVMs/tree/main/Data%20Set) comes in the form of a csv table. The columns most relevant to our task are 'text' and 'airline_sentiment'.
- Data must be represented as a [N x d] matrix, but what we have on our hands is unstructured text.
- The simplest solution to transform an airline review into a vector is bag of words. We maintain a global vocabulary of word patterns gathered from our corpus, with single words such as "great", "horrible", and optionally consecutive words (N-grams) like "friendly service", "luggage lost". Suppose we have already collected a total of 10000 such patterns, to transform a sentence into a 10000-dimensional vector, we simply scan it and look for the patterns that appear and set their correponding entries to 1 and leave the rest at 0. What we end up with is a sparse vector that can be fed into SVMs.
- The data is not balanced, with siginificant more negatives than neutral + positives. Therefore we group neutral and positive into one category and the final ratio of non-negative vs negative is about 1:2. This is consistent across train, val and test.

# Procedure
- Set up the dataset.

```
import os.path as osp
import pandas as pd
import re, nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

from sklearn.feature_extraction.text import CountVectorizer

data_root = 'data/tweets'
!curl -O https://ttic.uchicago.edu/~nsm/ttic_31020_2020/hw_3/dataset/train.csv
!curl -O https://ttic.uchicago.edu/~nsm/ttic_31020_2020/hw_3/dataset/val.csv
!curl -O https://ttic.uchicago.edu/~nsm/ttic_31020_2020/hw_3/dataset/test_release.csv

data_root = ''
train, val, test = \
    pd.read_csv(osp.join(data_root, 'train.csv')), \
    pd.read_csv(osp.join(data_root, 'val.csv')), \
    pd.read_csv(osp.join(data_root, 'test_release.csv'))

print(train.head(3))

print(train.airline_sentiment.value_counts())
```
- Build the vocabulary and vector representations for each word.

```
stop_words = set(stopwords.words('english'))
wordnet_lemmatizer = WordNetLemmatizer()
def tokenize_normalize(tweet):
    only_letters = re.sub("[^a-zA-Z]", " ", tweet)
    tokens = nltk.word_tokenize(only_letters)[2:]
    lower_case = [l.lower() for l in tokens]
    filtered_result = list(filter(lambda l: l not in stop_words, lower_case))
    lemmas = [wordnet_lemmatizer.lemmatize(t) for t in filtered_result]
    return lemmas

# the sklearn vectorizer scans our corpus, build the vocabulary, and changes text into vectors
vectorizer = CountVectorizer(
    strip_accents='unicode', 
    lowercase=True, 
    tokenizer=tokenize_normalize,
    ngram_range=(1,1),  # you may want to try 2 grams. The vocab will get very large though,
    min_df=100,  # this parameter deletes words that occur in less than min_df
                # documents. decreasing this will increase the vocabulary size,
                # but may also increase the runtime.
)
# first learn the vocabulary
vectorizer.fit(pd.concat([train, val]).text)


print( list(vectorizer.vocabulary_.items())[:10] )
print("\n vocabulary size {}".format(len(vectorizer.vocabulary_)))
```
- Set up the training set and prepare the training data so that we can call SVM as a black-box.

```
X = {}
y = {}
X['train'] = vectorizer.transform(train.text).toarray()
X['val'] = vectorizer.transform(val.text).toarray()
X['test'] = vectorizer.transform(test.text).toarray()

# note that our data is 10250 dimensional. 
# This is a little daunting for laptops and coming up with a manageable vector
# representation is a major topic in Natural Language Processing.
print(X['train'].shape)

# convert the word labels of 'positive', 'neutral', 'negative' into integer labels
# note that positive and neural belong to one category, labelled as 1, while negative stands alone as the other
for name, dataframe in zip(['train', 'val'], [train, val]):
    sentiments_in_words = dataframe['airline_sentiment'].tolist()
    int_lbls = np.array( list(map(lambda x: -1 if x == 'negative' else 1, sentiments_in_words)), dtype=np.int32 )
    y[name] = int_lbls
```
- Use Linear SVM, Kernel SVM with linear kernel and Kernel SVM with Gaussian kernel on the airline dataset。

- Linear SVM

```
svm = LinearSVM(C=1000)
svm.fit(X['train'], y['train'],lr_sched=lambda t: 1/(.1*t), num_epochs=10)
compute_acc(svm, X['train'], y['train'])
compute_acc(svm, X['val'], y['val'])
```
  * Kernel SVM with linear kernel

```
svm = KernelSVM(Kernel.linear(), C=100)
svm.fit(X['train'].astype(float), y['train'].astype(float))
compute_acc(svm, X['train'], y['train'])
compute_acc(svm, X['val'], y['val'])
```
  * Kernel SVM with Gaussian kernel

```
svm = KernelSVM(Kernel.gaussian(sigma=1), C=10)
svm.fit(X['train'].astype(float), y['train'].astype(float))
compute_acc(svm, X['train'], y['train'])
compute_acc(svm, X['val'], y['val'])
```

- Generate test output.

# Prediction Accuracy:

## Linear SVM

![image](https://user-images.githubusercontent.com/95513386/146266781-ba1eaff3-5abf-4864-a9a9-bfaa290137d9.png)

## Kernel SVM with linear kernel

![image](https://user-images.githubusercontent.com/95513386/146266830-700dd54c-6f7f-4dfa-bf28-125508fa0639.png)

## Kernel SVM with Gaussian Kernel

![image](https://user-images.githubusercontent.com/95513386/146266861-7df75e9c-f073-4d8a-8443-ae4987b11491.png)

We can see that Linear SVM is the best to train airline review data with the highest accuracy.
