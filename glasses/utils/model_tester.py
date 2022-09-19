from typing import Callable, Dict, Optional, Tuple, TypedDict

import torch
from torch import Tensor, nn

from glasses.config import Config
from glasses.models.vision.image.classification.outputs import (
    ModelForImageClassificationOutput,
)


def model_for_classification_output_test(
    model_output_dict: ModelForImageClassificationOutput,
):
    logits = model_output_dict["logits"]
    logits.mean().backward()


model_output_test_strategies = {
    ModelForImageClassificationOutput: model_for_classification_output_test
}


def create_input_dict(input_shape_dict: Dict[str, Tuple[int]]) -> Dict[str, Tensor]:
    input_dict = {}
    for name, shape in input_shape_dict.items():
        input_dict[name] = torch.randn(shape)
    return input_dict


def check_model_output_dict_shape(
    model_output_dict: Dict[str, Tensor], output_shape_dict
):
    for name, shape in output_shape_dict.items():
        assert model_output_dict[name].shape == shape


def check_grad_exists(model: nn.Module):
    for name, param in model.named_parameters():
        assert param.grad is not None, f"grad for param = {name} doesn't exist."


def check_output_equivalence(
    model_output_dict: Dict[str, Tensor], output_equivalence_dict: Dict[str, Tensor]
):
    for name, value in output_equivalence_dict.items():
        assert torch.allclose(
            model_output_dict[name], value, atol=1e-5
        ), f"param = {name} doesn't match equivalance."


def model_tester(
    config: Config,
    input_dict: Dict[str, Tensor],
    output_shape_dict: Dict[str, Tuple[int]],
    output_dict_type: TypedDict,
    output_test_strategy: Optional[Callable[[TypedDict], None]] = None,
    output_equivalence_dict: Optional[Dict[str, Tensor]] = None,
):
    """
    This functions tests the mdoel using different checks. In order:

    - check that model's outputs shapes matched `output_shape_dict`
    - check that the model can be trained, we use `output_dict_type` to know how to run output's specific tests. For example, for image classification you should pass `output_dict_type=ModelForImageClassificationOutput` and we will try to run `outputs["logits"].mean().backward()`
    - optionally, if `output_equivalence_dict` we will check that model's outputs match

    Args:
        config (Config): Configuration we will use to build the model.
        input_dict (Dict[str, Tensor]): Dictionary containing the inputs for the model. E.g. `{ "pixel_values" : torch.randn((1, 3, 224, 224))}`
        output_shape_dict (Dict[str, Tuple[int]]): Dictionary containing the expected shaped in the output. E.g. ` {"logits": (1, 1000)}`
        output_dict_type (TypedDict): The type of output, e.g. ModelForImageClassificationOutput](). Used to now which test strategy to run, based on the type we know how to test it. Defaults to None.
        output_test_strategy (Optional[Callable[[TypedDict]]], optional): If passed, we will use this strategy instead.
        output_equivalence_dict (Optional[Dict[str, Tensor]], optional): If passes, we will check that the model's output are equal to the values inside it. Defaults to None.
    """
    with torch.no_grad():
        # we are able to create the model from a config
        model = config.build().eval()
        model_output_dict = model(**input_dict)
        check_model_output_dict_shape(model_output_dict, output_shape_dict)
    model = model.train()
    model_output_dict = model(**input_dict)
    # we are able to check that the model can be trained
    output_test_strategy = output_test_strategy
    if output_test_strategy is None:
        output_test_strategy = model_output_test_strategies[output_dict_type]
    output_test_strategy(model_output_dict)
    check_grad_exists(model)
    # we are able to check outputs equivalence if passed
    if output_equivalence_dict:
        with torch.no_grad():
            # here run the model with a known input
            check_output_equivalence(model_output_dict, output_equivalence_dict)
