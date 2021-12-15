---
title: Sample Post - Markdown Syntax
commentable: flase
Edit: 2021-12-15
mathjax: true
mermaid: true
tags: SVM Kernel
categories: Data
description: This is a post for sample code [SVMs](https://github.com/jyang-zhou/SVMs).
---

# SVMs
SVMs via sub-gradient descent and quadratic programming with sentiment analysis on tweets on US airline service quality. We focus on using Linear SVM, Kernel SVM with linear kernel and Kernel SVM with RBF kernel on the dataset.

{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "wCq8kJ42ghDl"
   },
   "source": [
    "### Sample Code for SVMs via sub-gradient descent and quadratic programming"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "cellView": "form",
    "id": "vNN6eZW5ghDm"
   },
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import numpy.random as npr\n",
    "\n",
    "import matplotlib.cm as cm\n",
    "import matplotlib.pyplot as plt\n",
    "%matplotlib inline\n",
    "\n",
    "from math import sqrt\n",
    "\n",
    "import csv\n",
    "import cvxopt\n",
    "\n",
    "def create_download_file(fname, preds):\n",
    "    \"\"\"Create file with predictions written as a csv file\n",
    "    \"\"\"\n",
    "    ofile  = open(fname, \"w\")  \n",
    "    writer = csv.writer(\n",
    "        ofile, delimiter=',', quotechar='\"', quoting=csv.QUOTE_ALL\n",
    "    )\n",
    "    writer.writerow(['id', 'category'])\n",
    "    for i in range(preds.shape[0]):\n",
    "        writer.writerow([i, preds[i]])\n",
    "\n",
    "\n",
    "def plot_decision_countour(svm, X, y, grid_size=100):\n",
    "    x_min, x_max = X[:, 0].min(), X[:, 0].max()\n",
    "    y_min, y_max = X[:, 1].min(), X[:, 1].max()\n",
    "    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_size),\n",
    "                         np.linspace(y_min, y_max, grid_size),\n",
    "                         indexing='ij')\n",
    "    data = np.stack([xx, yy], axis=2).reshape(-1, 2)\n",
    "    pred = svm.predict(data).reshape(xx.shape)\n",
    "    plt.contourf(xx, yy, pred,\n",
    "                 cmap=cm.Paired,\n",
    "                 levels=[-0.001, 0.001],\n",
    "                 extend='both',\n",
    "                 alpha=0.8)\n",
    "    flatten = lambda m: np.array(m).reshape(-1,)\n",
    "    plt.scatter(flatten(X[:,0][y==-1]),flatten(X[:,1][y==-1]),\n",
    "                  c=flatten(y)[y==-1],cmap=cm.Paired,marker='o')\n",
    "    plt.scatter(flatten(X[:,0][y==1]),flatten(X[:,1][y==1]),\n",
    "                  c=flatten(y)[y==1],cmap=cm.Paired,marker='+')\n",
    "    \n",
    "    plt.xlim(x_min, x_max)\n",
    "    plt.ylim(y_min, y_max)\n",
    "    plt.plot()\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "def test_SVM(svm, num_samples=500,linear=False):\n",
    "    \"\"\"test svm\n",
    "    \"\"\"\n",
    "    np.random.seed(783923)\n",
    "\n",
    "    X = npr.random((num_samples, 2)) * 2 - 1\n",
    "    if linear:\n",
    "      y = 2 * (X.sum(axis=1) > 0) - 1.0\n",
    "    else: \n",
    "      y = 2 * ((X ** 2).sum(axis=1) - 0.5 > 0) - 1.0\n",
    "    svm.fit(X,y)\n",
    "    \n",
    "    plot_decision_countour(svm, X, y)\n",
    "\n",
    "    from datetime import datetime\n",
    "    np.random.seed(int(round(datetime.now().timestamp())))\n",
    "\n",
    "def compute_acc(model, X, y):\n",
    "    pred = model.predict(X)\n",
    "    size = len(y)\n",
    "    num_correct = (pred == y).sum()\n",
    "    acc = num_correct / size\n",
    "    print(\"{} out of {} correct, acc {:.3f}\".format(num_correct, size, acc))"
   ]
  }
}

# Data Set

- The [data](https://github.com/jyang-zhou/SVMs/tree/main/Data%20Set) comes in the form of a csv table. The columns most relevant to our task are 'text' and 'airline_sentiment'.
- Data must be represented as a [N x d] matrix, but what we have on our hands is unstructured text.
- The simplest solution to transform an airline review into a vector is bag of words. We maintain a global vocabulary of word patterns gathered from our corpus, with single words such as "great", "horrible", and optionally consecutive words (N-grams) like "friendly service", "luggage lost". Suppose we have already collected a total of 10000 such patterns, to transform a sentence into a 10000-dimensional vector, we simply scan it and look for the patterns that appear and set their correponding entries to 1 and leave the rest at 0. What we end up with is a sparse vector that can be fed into SVMs.
- The data is not balanced, with siginificant more negatives than neutral + positives. Therefore we group neutral and positive into one category and the final ratio of non-negative vs negative is about 1:2. This is consistent across train, val and test.

# Procedure
- Set up the dataset.

- Build the vocabulary and vector representations for each word.
- Set up the training set and prepare the training data so that we can call SVM as a black-box.
- Use Linear SVM, Kernel SVM with linear kernel and Kernel SVM with RBF kernel on the airline dataset.
- Generate test output.

# Prediction Accuracy:

## Linear SVM

![image](https://user-images.githubusercontent.com/95513386/146266781-ba1eaff3-5abf-4864-a9a9-bfaa290137d9.png)

## Kernel SVM with linear kernel

![image](https://user-images.githubusercontent.com/95513386/146266830-700dd54c-6f7f-4dfa-bf28-125508fa0639.png)

## Kernel SVM with Gaussian Kernel

![image](https://user-images.githubusercontent.com/95513386/146266861-7df75e9c-f073-4d8a-8443-ae4987b11491.png)

We can see that Linear SVM is the best to train airline review data with the highest accuracy.

