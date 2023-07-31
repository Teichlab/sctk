# 🔬 Developer Guide

Thank you for considering to contribute to the project! This guide will help you
to get started with the development of the project. If you have any questions,
please feel free to ask them in the issue tracker.

## Testing

The project uses [pytest](https://docs.pytest.org/en/stable/) for testing. To
run the tests, please run `pytest` in the root directory of the project. Please
make sure that you add tests for your code before submitting a pull request.

The existing tests can also help you to understand how the code works. If you
have any questions, please feel free to ask them in the issue tracker.

**Before submitting a pull request, please make sure that all tests pass.**

## Small Contributions

If you want to contribute a small change (e.g. a bugfix), you can probably
immediately go ahead and create a pull request. For more substantial changes or
additions, please read on.

## Larger Contributions

If you want to contribute a larger change, please create an issue first. This
will allow us to discuss the change and make sure that it fits into the project.
It can happen that development for a feature is already in progress, so it is
important to check first to avoid duplicate work. If you have any questions,
feel free to approach us in any way you like.

## Versioning

We use [semantic versioning](https://semver.org/) for the project. This means
that the version number is incremented according to the following scheme:

- Increment the major version number if you make incompatible API changes.

- Increment the minor version number if you add functionality in a backwards-
  compatible manner.

- Increment the patch version number if you make backwards-compatible bug fixes.

You can use the `bumpversion` tool to update the version number in the
`pyproject.toml` file. This will create a new git tag automatically.

```
bumpversion [major|minor|patch]
```