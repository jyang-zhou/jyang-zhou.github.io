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

```
function test() {
  console.log("notice the blank line before this function?");
}
```

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

