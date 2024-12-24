[![PyPI version fury.io](https://badge.fury.io/py/sklearn_minisom.svg)](https://pypi.org/project/sklearn_minisom/)
[![Downloads](https://static.pepy.tech/personalized-badge/sklearn_minisom?period=total&units=international_system&left_color=black&right_color=blue&left_text=Downloads)](https://pepy.tech/project/sklearn_minisom)

# sklearn_minisom

MiniSom is Numpy based implementation of the Self Organizing Maps (SOM). SOM is a type of Artificial Neural Network able to convert complex, nonlinear statistical relationships between high-dimensional data items into simple geometric relationships on a low-dimensional display. Minisom is designed to allow researchers to easily build on top of it and to give students the ability to quickly grasp its details.

This is MiniSom library wrapper for seamless integration with SciKit-learn package.

Credits to:
* Wrapped library: [MiniSom by Giuseppe Vettigli](https://github.com/JustGlowing/minisom)

* SciKit-learn: [Docs](https://scikit-learn.org/stable/) [Github](https://github.com/scikit-learn/scikit-learn)

This wrapper aims to integrate MiniSOM library into SciKit-learn ecosystem.
It enables easy integration with Scikit-learn pipelines and 
tools like GridSearchCV for hyperparameter optimization. It also provides easy, scikit-learn like API for developers to interact with while aiming to sustain high flexibility and capabilities of MiniSom library.

Example clustering datasets from [Comparation of clustering algorithms on SciKit-learn](https://scikit-learn.org/stable/modules/clustering.html)

This is separate project and not part of MiniSom library due to creator's of the original project aim to keep their as lightweight as possible. 


### Table of content
- [Installation](#installation)
- [Examples](#Examples)

## Installation

Just use pip:

    pip install sklearn_minisom

Dependencies:
* minisom>=2.3.3
* scikit-learn
* numpy
* scipy
* pytest


## Examples

Just use it like any other scikit-learn cluster algorithm.

Let's start with importing required libraries and dataset.

```python
from sklearn.datasets import load_wine
from sklearn_minisom import MiniSOM
from sklearn.preprocessing import StandardScaler

data = load_wine()
X = data.data
X = StandardScaler().fit_transform(X)
```

You can use fit and predict separately.
```python
som = MiniSOM(3, 1, random_seed=40)
som.fit(X)
y = som.predict(X)
```

Or simply use convenient function.
```python
som = MiniSOM(3, 1, random_seed=40)
y = som.fit_predict(X)
```


Alternatively you can also use SciKit-learn pipelines.
```python
from sklearn.pipeline import Pipeline

pipeline = ([
    ('scaler', StandardScaler()),
    ('classifier', MiniSOM(3, 1, random_seed=40))
])

y = pipeline.fit_predict(X)
```
Now let's take a look at what we've got.

