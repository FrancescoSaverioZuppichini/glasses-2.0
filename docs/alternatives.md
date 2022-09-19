What inspired **Glasses**, how it compares to other alternatives and what it learned from them.

## Intro
**Glasses** wouldn't exist if not for the previous work of others and the **f* amazing open-source community**. We love every single one of you 🥰.



### <a href="https://github.com/huggingface/transformers" class="external-link" target="_blank">Transformers</a> 

`Transformers` is not a model toolbox, therefore you cannot compose and share individual building components. Moreover, their philosophy of one model one file creates a lot of code repetition creating huge model files hard to read and understand.

Their testing approach, even if more robust than ours, increase the development time due to the amount of work required to make the model pass the tests.

Moreover, we are not motivated by financial gains. Thus, we don't follow the hype blindly.

!!! check "Inspired **Glasses** to"
   How to call the `AutoModel` class I guess.

### <a href="https://airctic.com/0.12.0/" class="external-link" target="_blank">IceVision</a> 

`IceVision` is a great library to train models in different vision tasks. The main difference between glasses is that they don't "own" the model, they rely on third-party libraries with custom adapters.

This strategy makes it easier to increase the pool of available models, but a minor change in one dependency may break the whole codebase. 

Our goal is also *to teach*, for this reason, we implemented all the models we use in a (hopefully) clear and concise way.

!!! check "Inspired **Glasses** to"
    Provide a simple way to defined tasks' inputs/outsputs.


### <a href="https://github.com/open-mmlab" class="external-link" target="_blank">OpenMMLab</a> 


`OpenMMLab` team has different amazing libraries for each vision task. However, they are fragmented and not easy to use due to the configuration system.

Their configuration system is not typed, therefore is impossible for the end-user to know what to place inside it. They used inheritance in configuration, making it challenging to have a full view of the system.

Finally, their `Trainer` is a closed box; very hard to extend. We will rely on [Lightning](https://www.pytorchlightning.ai/)


!!! check "Inspired **Glasses** to"
    Create nice and composable configurations.


### <a href="https://github.com/facebookresearch/detectron2" class="external-link" target="_blank">Detectron2</a> 

If you know how use it, you are my hero.