# Changelog
All notable changes to this project will be documented in this file.

This project uses [Semantic Versioning][sv].

## [Unreleased][new]

### Added
- Add a function to initialize weights and biases for all layers of the neural
  network.
- Add `linear_forward()`to compute the linear part of forward propagation for a
  given layer.
- Add `layer_forward()` to compute forward propagation through a single layer.
- Add `sigmoid()` and `relu()` activation functions.
- Add `model_forward()` to compute forward propagation for the complete model
  comprising (L-1) ReLU units followed by a single, sigmoid unit.
- Add `compute_cost()` to calculate the cross-entropy cost from the model
  predictions.

### Changed

## [0.0.0][0.0.0] — 2020-03-15

### Added
- Create the project. A library for building and training deep neural networks
  in Python.

---
_This file is composed with [GitHub Flavored Markdown][gfm]._

[gfm]: https://github.github.com/gfm/
[sv]: https://semver.org

[new]: https://github.com/petejh/deepen/compare/HEAD..v0.0.0
[0.0.0]: https://github.com/petejh/deepen/releases/tag/v0.0.0
