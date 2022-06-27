from glasses.config import Config
from typing import Tuple, Dict, Optional, TypedDict
from torch import Tensor, nn
import torch
from glasses.models.vision.classification.outputs import ModelForClassificationOutput


def model_for_classification_output_test(
    model_output_dict: ModelForClassificationOutput,
):
    logits = model_output_dict["logits"]
    logits.mean().backward()


model_output_test_strategies = {
    ModelForClassificationOutput: model_for_classification_output_test
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
    output_equivalence_dict: Optional[Dict[str, Tensor]] = None,
):
    with torch.no_grad():
        # we are able to create the model from a config
        model = config.build().eval()
        model_output_dict = model(**input_dict)
        check_model_output_dict_shape(model_output_dict, output_shape_dict)
    model = model.train()
    model_output_dict = model(**input_dict)
    # we are able to check that the model can be trained
    model_output_test_strategies[output_dict_type](model_output_dict)
    check_grad_exists(model)
    # we are able to check outputs equivalence if passed
    if output_equivalence_dict:
        # here run the model with a known input
        # [TODO]
        check_output_equivalence(model_output_dict, output_equivalence_dict)
