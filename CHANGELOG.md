# Changelog
All notable changes to this project will be documented in this file.

This project uses [Semantic Versioning][sv].

## [Unreleased][new]

### Added
- Add `relu_backward()` to compute backward propagation through a ReLU unit.
- Add `sigmoid_backward()` to compute backward propagation through a sigmoid unit.
- Add `linear_backward()` to compute the linear part of backward propagation for
  a given layer.
- Add `layer_backward()` to compute backward propagation for a single layer.

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

[new]: https://github.com/petejh/deepen/compare/HEAD..v0.1.0
[0.1.0]: https://github.com/petejh/deepen/releases/tag/v0.1.0
[0.0.0]: https://github.com/petejh/deepen/releases/tag/v0.0.0
