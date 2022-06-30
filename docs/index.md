# Glasses 😎

!!! important "A long way to go"
    🚧 This project is WIP. We will make it perfect, but we are not still there! If you want to help out, check the [Contributing Guide](/contributing.md) 💜.

A compact, concise, and customizable deep learning library. This library currently supports deep learning models for computer vision.

*Glasses* is a model toolbox to make it easier for **everybody** to use, learn and share deep learning models.


**Documentation**: [TODO](TODO)

**Source Code**: [https://github.com/FrancescoSaverioZuppichini/glasses](https://github.com/FrancescoSaverioZuppichini/glasses)


## TL;DR

This library has

- human readable code, no *research code*
- common component are shared across [models](#Models)
- same APIs for all models (you learn them once and they are always the same)
- clear and easy to use model constomization (see [here](#block))
- [classification](#classification) and [segmentation](#segmentation) 
- easy to contribute, see the [contribution guide](/contributing)
- emoji in the name ;)

## Requirements

Python 3.8+
## Installation

<div class="termy">

```console
$ pip install git+https://github.com/FrancescoSaverioZuppichini/glasses.git
---> 100%
```

</div>

## Motivations
Almost all existing implementations of the most famous model are written with very bad coding practices, what today is called research code. We struggled to understand some of the implementations even if in the end were just a few lines of code.

Most of them are missing a global structure, they used tons of code repetition, and they are not easily customizable and not tested. Thus, not easy to share and use by everybody.



Head over the [getting started](getting_started) guide

## RoadMap

We plan to have three main steps in the development

* **Models**: Defined different models for different tasks, the configuration system, and how to save/load them. ⬅️ **We are here!**
* **Tasks**: Defined the train/evaluation lifecycle for each task. 
* **Pipelines**: Defined the whole lifecycle for a task, from data to training.

  
## Contributing
Please contribute using [GitHub Flow](https://docs.github.com/en/get-started/quickstart/github-flow). Create a branch, add commits, and open a pull request.

Please read [contributing](/docs/contributing.md) for details on our CODE OF CONDUCT, and the process for submitting pull requests to us.


## License¶
This project is licensed under the terms of the MIT license.