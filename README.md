# sklearn_minisom

MiniSom library wrapper for seamless integration with SciKit-learn library.

Credits to:
* Wrapped library: [MiniSom by Giuseppe Vettigli](https://github.com/JustGlowing/minisom)

* SciKit-learn: [Docs](https://scikit-learn.org/stable/) [Github](https://github.com/scikit-learn/scikit-learn)

This wrapper aims to integrate MiniSOM library into SciKit-learn ecosystem.
It enables easy integration with Scikit-learn pipelines and 
tools like GridSearchCV for hyperparameter optimization. It also provides easy, scikit-learn like API for developers to interact with while aiming to sustain high flexibility and capabilities of MiniSom library.

It is separate project and not part of MiniSom library due to creator's of the original project aim to keep their as lightweight as possible. 

- [Installation](#installation)
- [Examples](#Examples)

## Installation

Just use pip:

    pip install sklearn_minisom


## Examples

Predict Iris Dataset clusters.

```python
from sklearn_minisom import MiniSOM
from sklearn import datasets

iris = datasets.load_iris()
iris_data = iris.data

som = MiniSOM()
som.fit(iris_data)

y = som.predict(iris_data)
print(y)
```

Transform Iris Dataset data.
```python
from sklearn_minisom import MiniSOM
from sklearn import datasets

iris = datasets.load_iris()
iris_data = iris.data

som = MiniSOM()

som.fit_transform(iris_data)
```


Use to build SciKit-learn pipelines
```python
from sklearn_minisom import MiniSOM
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

X = [[ 0.80,  0.55,  0.22,  0.03],
        [ 0.82,  0.50,  0.23,  0.03],
        [ 0.80,  0.54,  0.22,  0.03],
        [ 0.80,  0.53,  0.26,  0.03],
        [ 0.79,  0.56,  0.22,  0.03],
        [ 0.75,  0.60,  0.25,  0.03],
        [ 0.77,  0.59,  0.22,  0.03]]  

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', MiniSOM(x=10, y=5, sigma=1, random_seed=42))
    ])

y = pipeline.fit_predict(X)
print(y)
```

