# Deepen
A library for building and training deep neural networks in Python.

## Motivation
Deepen is my first project in [Python][python]; in fact, I started it principally to teach
myself the Python language. The subject itself grew out of self-paced learning
about deep neural networks, including a [Coursera][coursera] class from
[deeplearning.ai][deepai], which I unreservedly recommend.

Deepen implements a neural network from "first principles", building the forward
and backward propagation processes out of elementary differential calculus. And
while I can't recommend it for production use (you should prefer [Keras][keras],
[scikit-learn][scikit], [TensorFlow][tensor], or similar libraries for those applications), it is
nevertheless accurate and efficient. It is entirely suitable for educational
purposes and exploration of deep learning topics.

"Deepen" is a pretty transparent pun on "deep neural network (DNN)". But is also
stands for the way I return to the project from time to time to bear down on
various concepts in idiomatic Python, tooling, and packaging. Among others, I've
experimented with:

- A blend of _in-situ_ code documentation styles modeled after [NumPy][numpy] and [SciPy][scipy].
- List comprehensions
- Generator functions built around `yield`.
- State encapsulation and validation using `@property` decorators.
- Context blocks keyed by `with`.
- [NumPy][numpy] for linear algebra.
- [H5][h5py] storage for both data and model serialization.
- A marriage of object-oriented and functional design.
- Structuring a library for distribution on [PyPi][pypi].
- Dependency management and packaging with [Poetry][poetry], replacing `setuptools`.
- Automated unit testing with `unittest` and [pytest][pytest], including mocks, spies, and test groups.
- Managing isolated testing environments with [tox][tox].

Note the code is not thread-safe, as a whole, as the results for all layers have
to be collated after each pass. However, the layer calculations themselves are
functionally independent, and do not depend on shared state. For very large
layers or datasets, calculations within each pass could be multi-threaded up to
the collation point.

## Installation
```bash
~$ pip install git+https://github.com/petejh/deepen.git
```

## Usage
```python
import numpy as np
from deepen.model import Model

# For a single hidden layer comprising 10 nodes:
dnn = Model(layer_dims=[1, 10, 1], learning_rate=0.08)

# ... load data
# ... normalize data
# ... partition data

# Assuming `n` RGB images of `x` by `y` pixels:
# assert(train_x.shape == (x * y * 3, n))
# assert(train_y.shape == (1, n))

params, cost = dnn.learn(train_x, train_y)
predict_y = dnn.predict(test_x)
accuracy = np.sum(predict_y == test_y) / test_y.shape[1]

dnn.save('/path/to/data/dnn.h5')
```

You can load a trained model from storage:
```python
dnn = Model()
dnn.load('/path/to/data/dnn.h5')
```

## Contributing
Deepen provides a safe, welcoming space for collaboration. Everyone
contributing to our project—including the codebase, issue trackers, chat, email,
social media and the like—is expected to uphold our [Code of Conduct][coc].

Bug reports and pull requests are welcome on [GitHub][orig].

To contribute code, first [fork the project][fork] on GitHub and make a local
clone. Create a topic branch, make and commit your changes, and push this
branch back to GitHub. [Submit a pull request][pull] and ask to have your work
reviewed for inclusion in the main repository.

### Setup
Deepen uses [Poetry][poetry] and [tox][tox] for package management and automated
testing, respectively. Both packages are meant to be used as standalone tools,
rather than project-specific dependencies; you will want to install them globally
for the active `python` version:
```bash
~$ pip install --global poetry
~$ pip install --global tox
```

You can simply run tox to execute all tests in pre-defined, isolated environments:
```bash
~$ tox
```

Activate the virtual environment created by Poetry with:
```bash
~$ source `poetry env info --path`/bin/activate
```

## License
This project is available as open source under the terms of the [MIT License][mit].

---
_This file is composed with [GitHub Flavored Markdown][gfm]._

[coc]: https://github.com/petejh/deepen/blob/master/CODE_OF_CONDUCT.md
[coursera]: https://www.coursera.org/specializations/deep-learning
[deepai]: https://www.deeplearning.ai
[fork]: https://help.github.co://help.github.com/en/github/getting-started-with-github/fork-a-repo
[gfm]: https://github.github.com/gfm/
[h5py]: https://www.h5py.org
[keras]: https://keras.io
[orig]: https://github.com/petejh/deepen
[mit]: https://github.com/petejh/deepen/blob/master/LICENSE.txt
[numpy]: https://numpy.org
[poetry]: https://python-poetry.org
[pull]: https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request-from-a-fork/
[pypi]: https://pypi.org
[pytest]: https://pytest.org
[python]: https://www.python.org
[scikit]: https://scikit-learn.org
[scipy]: https://www.scipy.org
[tensor]: https://www.tensorflow.org
[tox]: https://tox.readthedocs.io/en/latest/
