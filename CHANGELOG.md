# Changelog
All notable changes to this project will be documented in this file.

This project uses [Semantic Versioning][sv].

## [Unreleased][new]

## Added
- [Poetry][poetry] for package configuration and dependency management.
- [pytest][pytest] for unit testing.
- [tox][tox] for automated testing in isolated environments.

## Changed
- Restructured project with a `src-folder` layout to better accommodate
  distribution as a library
- Expanded `README.md` with Motivation, Usage, and Contributing/Setup details.

## Removed
- `setuptools` artifacts, including `setup.py` and `requirements.txt`

## [0.4.0][0.4.0] — 2020-05-29

### Added
- Add `learn_generator()` to provide stepwise learning for the model, with
  updated parameters and cost returned at a given interval of propagation cycles.
- Add `save()` and `load()` to serialize/deserialize the model to a file.

### Changed
- `learn()` returns a list of (params, cost) with one tuple for each complete
  cycle of forward and back propagation.

## [0.3.0][0.3.0] — 2020-04-23

### Added
- Create a Model class with `learn()` to train the model and `predict()` to
  transform input data using the trained model.
- Create the `deepen.propagation` module.
- Add `requirements.txt` and indicate a dependency on `numpy` in `setup.py`.

### Changed
- Move functions supporting feedforward processing and backpropagation to the
  new `deepen.propagation` module.
- Weights are initialized to have standard deviation of ≈ 1.0 in each layer.
- Rewrite the computation of the cost function for greater clarity.

## [0.2.0][0.2.0] — 2020-03-30

### Added
- Add `relu_backward()` to compute backward propagation through a ReLU unit.
- Add `sigmoid_backward()` to compute backward propagation through a sigmoid unit.
- Add `linear_backward()`, `layer_backward()`, and `model_backward()` to compute
  backward propagation for the complete model comprising (L-1) ReLU units
  followed by a single, sigmoid unit.
- Add `update_params()` to calculate new weights and biases for each layer after
  a complete forward and backward pass through the model.
- Add unit tests for all new functions.

### Changed

## [0.1.0][0.1.0] — 2020-03-26

### Added
- Add a function to initialize weights and biases for all layers of the neural
  network.
- Add `linear_forward()`, `layer_forward()`, and `model_forward()` to compute
  forward propagation for the complete model comprising (L-1) ReLU units
  followed by a single, sigmoid unit.
- Add `compute_cost()` to calculate the cross-entropy cost from the model
  predictions.
- Add the `deepen.activations` module with `sigmoid()` and `relu()` activation
  functions.
- Add unit tests for `deepen.module`.
- Add unit tests for `deepen.activations`.

## [0.0.0][0.0.0] — 2020-03-15

### Added
- Create the project. A library for building and training deep neural networks
  in Python.

---
_This file is composed with [GitHub Flavored Markdown][gfm]._

[gfm]: https://github.github.com/gfm/
[sv]: https://semver.org

[new]: https://github.com/petejh/deepen/compare/HEAD..v0.4.0
[0.4.0]: https://github.com/petejh/deepen/releases/tag/v0.4.0
[0.3.0]: https://github.com/petejh/deepen/releases/tag/v0.3.0
[0.2.0]: https://github.com/petejh/deepen/releases/tag/v0.2.0
[0.1.0]: https://github.com/petejh/deepen/releases/tag/v0.1.0
[0.0.0]: https://github.com/petejh/deepen/releases/tag/v0.0.0
