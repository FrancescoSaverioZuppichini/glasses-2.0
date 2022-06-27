# Glasses 😎

!!! important
    🚧 This project is WIP. We will make it perfect, but we are not still there! If you want to help out, check the [Contributing Guide](/contributing.md) 💜.

A Compact, concise and customizable deep learning computer library. Currently focused on vision.

*Glasses* is a model toolbox with the goal of making easier for **everybody** to use, learn and share deep learning models.


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
Almost all existing implementations of the most famous model are written with very bad coding practices, what today is called research code. I struggled to understand some of the implementations even if in the end were just a few lines of code.

Most of them are missing a global structure, they used tons of code repetition, they are not easily customizable and not tested. Thus, not easy to share and use by everybody.

### Comparison with other libraries

Where does *glasses* stand across this amazing work of open source?

* [Transformers](https://github.com/huggingface/transformers) is not a model toolbox, therefore you cannot compose and share individual building components. Moreover, their phisolofy of one model one file creates a lot of code repetition creating huge model files hard to read and to understand.

    Their testing approach, even if more robust than ours, increase the development time due to the amount of work required to make the model pass the tests.

    Moreover, we are not motivated by financial gains. Thus, we don't follow the hype blindy.


* [IceVision](https://airctic.com/0.12.0/) is a great library to train models in different vision tasks. The main different between glasses is that they don't "own" the model, they rely on third party libraries with custom adapters.

    This strategy makes it easier to increase the pool of available models, but a minor change in one dependencies may break the whole codebase. 

    Our goal is also *to teach*, for this reason we implemented all the model we use in a (hopefully) clear and conside way.


* [OpenMMLab](https://github.com/open-mmlab) team has different amazing libraries for each vision task. However, they are fragmented and not easy to use due to the configuration system.

    Their configuration system is not typed, therefore is impossible for the end user to know what to place inside it. They used inheritance in configuration, making it really challanging to have a full view of the system.

    Finally, their trained is a close box; very hard to extend. We will rely on [Lightning](https://www.pytorchlightning.ai/)


* [Detectron2](https://github.com/facebookresearch/detectron2). If you were able to use it, you are my hero.



Head over the [getting started](getting_started) guide

## RoadMap

We plan to have three main steps in development

* **Models**: Definied different models for different tasks, the configuration system and how to save/load them. ⬅️ **We are here!**
* **Tasks**: Definied the train/evaluation lifecycle for each task. 
* **Pipelines**: Definied the whole lifecyle for a task, from data to train.

  