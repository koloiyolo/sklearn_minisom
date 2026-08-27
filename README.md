[![PyPI version fury.io](https://badge.fury.io/py/sklearn_minisom.svg)](https://pypi.org/project/sklearn_minisom/)
[![Downloads](https://static.pepy.tech/personalized-badge/sklearn_minisom?period=total&units=international_system&left_color=black&right_color=blue&left_text=Downloads)](https://pepy.tech/project/sklearn_minisom)

# sklearn_minisom

MiniSom is Numpy based implementation of the Self Organizing Maps (SOM). SOM is a type of Artificial Neural Network able to convert complex, nonlinear statistical relationships between high-dimensional data items into simple geometric relationships on a low-dimensional display. Minisom is designed to allow researchers to easily build on top of it and to give students the ability to quickly grasp its details.

This is MiniSom library wrapper for seamless integration with SciKit-learn package.

Credits to:

- Wrapped library: [MiniSom by Giuseppe Vettigli](https://github.com/JustGlowing/minisom)

- SciKit-learn: [Docs](https://scikit-learn.org/stable/) [Github](https://github.com/scikit-learn/scikit-learn)

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

```
pip install sklearn_minisom
```

or `uv`:

```
uv add sklearn_minisom
```

Dependencies:

- minisom>=2.3.6
- scikit-learn
- numpy

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

## Contributing

This project follows [Google Code Style] guidelines.
There is already configured `pre-commit` hook which is also part of CI. Before commiting please make sure you have `pre-commit` installed and run:

```bash
pre-commit install
```

After that, formatter and basic linters will be run automatically before each commit.

[Google Code Style]: https://google.github.io/styleguide/
