[![PyPI version fury.io](https://badge.fury.io/py/sklearn_minisom.svg)](https://pypi.org/project/sklearn_minisom/)
[![Downloads](https://static.pepy.tech/personalized-badge/sklearn_minisom?period=total&units=international_system&left_color=black&right_color=blue&left_text=Downloads)](https://pepy.tech/project/sklearn_minisom)

# sklearn_minisom

MiniSom is Numpy based implementation of the Self Organizing Maps (SOM). SOM is a type of Artificial Neural Network able to convert complex, nonlinear statistical relationships between high-dimensional data items into simple geometric relationships on a low-dimensional display. Minisom is designed to allow researchers to easily build on top of it and to give students the ability to quickly grasp its details.

This is MiniSom library wrapper for seamless integration with SciKit-learn package.

Credits to:
* Wrapped library: [MiniSom by Giuseppe Vettigli](https://github.com/JustGlowing/minisom)

* SciKit-learn: [Docs](https://scikit-learn.org/stable/) [Github](https://github.com/scikit-learn/scikit-learn)

This wrapper aims to integrate SklearnMinisom library into SciKit-learn ecosystem.
It enables easy integration with Scikit-learn pipelines and 
tools like GridSearchCV for hyperparameter optimization. It also provides easy, scikit-learn like API for developers to interact with while aiming to sustain high flexibility and capabilities of MiniSom library.

Example clustering datasets from [Comparation of clustering algorithms on SciKit-learn](https://scikit-learn.org/stable/modules/clustering.html)

![image](https://github.com/user-attachments/assets/9bf00573-1dee-455b-bd24-632b16dbec0b)

This is separate project and not part of MiniSom library due to creator's of the original project aim to keep their as lightweight as possible. 


### Table of content
- [Installation](#installation)
- [Examples](#examples)
- [Overview](#overview)

## Installation

Just use pip:

    pip install sklearn_minisom

Dependencies:
* minisom>=2.3.3
* scikit-learn
* numpy

## Examples

Just use it like any other scikit-learn cluster algorithm.

Let's start with importing required libraries and dataset.

```python
from sklearn.datasets import load_wine
from sklearn_minisom import SklearnMinisom
from sklearn.preprocessing import StandardScaler

data = load_wine()
X = data.data
X = StandardScaler().fit_transform(X)
```

You can use fit and predict separately.
```python
som = SklearnMinisom(3, 1, random_seed=40)
som.fit(X)
y = som.predict(X)
```

Or simply use convenient function.
```python
som = SklearnMinisom(3, 1, random_seed=40)
y = som.fit_predict(X)
```


Alternatively you can also use SciKit-learn pipelines.
```python
from sklearn.pipeline import Pipeline

pipeline = ([
    ('scaler', StandardScaler()),
    ('classifier', SklearnMinisom(3, 1, random_seed=40))
])

y = pipeline.fit_predict(X)
```
Now let's take a look at what we've got.

![image-1](https://github.com/user-attachments/assets/2111a12b-8e0f-453d-83d2-cb029465f112)

## Overview

| Class | Description |
| -------- | -------------- |
| SklearnMinisom() | Main class of the library, used to initialize the Self-Organizing Maps algorithm|

| Parameters | Description|
| -------- | -------------- |
| x : int  | x dimension of the SOM.|
| y : int  | y dimension of the SOM.|
| sigma : float | Spread of the neighborhood function.|
| learning_rate : float  | Initial learning rate.|
| num_iteration : int  | Number of iterations.|
| decay_function : string or callable | Function that reduces learning_rate at each iteration.|
| neighborhood_function : string | Function that weights the neighborhood of a position in the map.|
| topology : string  | Topology of the map.|
|activation_distance : string | Distance used to activate the map.|
|  random_seed : int  | y dimension of the SOM.|
| verbose : bool  | If True the status of the training will be printed each time the weights are updated.|
| use_epochs : bool  |  If True the SOM will be trained for num_iteration epochs.|
| fixed_points : dict  |A dictionary k : (c_1, c_2), that will force the training algorithm to use the neuron with coordinates (c_1, c_2) as winner for the sample k instead of the best matching unit.|
| sigma_decay_function : string  |  Function that reduces sigma at each iteration.|

| Attributes | Description|
| -------- | -------------- |
| labels_ : ndarray of shape (n_samples,)  | Labels of each point.|
| weights_ : ndarray of shape (grid_size_x, grid_size_y, feature_size) | Returns the weights of the neural network.|
| n_features_in_ : int | Number of features seen during fit.|
| inertia_ : float  | Sum of squared distances of samples to their closest neuron weight vector.|

| Functions | Description|
| -------- | -------------- |
| fit(X)  | Initializes SOM algorithm from minisom library and fits on it data matrix.|
| transform(X)  | Transform the data by finding the best matching unit (BMU) for each sample. Returns the BMU coordinates for each input sample.|
| predict(X) | Predict the cluster assignment (BMU) for each sample. Returns the grid position (BMU) for each sample as the cluster assignment.|
| fit_transform(X) | Fit the SOM and return the transformed BMU coordinates for each input sample.|
| fit_predict(X) | Fit the SOM and return the predicted cluster assignments.|
| get_params(deep=True)  | Get parameters of the estimator.|
| set_params(**params) | Set the parameters of this estimator. |
