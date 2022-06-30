from pathlib import Path
from typing import Dict

from cookiecutter.generate import generate_files
from rich import print_json
from typer import confirm

from cli.questions import Ask, Choices, IfHasSelected, Questions, Select

# This is how the input collected from the user looks like
# {
#   "model_name_lower_case": "my_model",
#   "ModelNameCammelCase": "MyModel",
#   "model_checkpoint": "my_model_base",
#   "modality": {
#     "vision": {
#       "type": "model",
#       "task": "image-classification"
#     }
#   }
# }

GLASSES_DIR = Path("./glasses").absolute()
TEMPLATE_DIR = Path("./templates").absolute()


def get_base_class_from_context(context: Dict) -> str:
    # a mapping that mimic the structure of the questions
    base_classes = {
        "vision": {
            "neck": "Neck",
            "backbone": "Backbone",
            "model": {"image-classification": "ModelForImageClassification"},
            "head": {"image-classification": "HeadForImageClassification"},
        }
    }

    # we need to find out which base class to use, let's get the modality e.g. `vision`
    modality = list(context["modality"].keys())[0]
    # now from the base_classes dict, use the modality to get it's base_class
    base_class = base_classes[modality][context["modality"][modality]["type"]]
    # if what we have found is a dict, it means that it can have tasks, we need to find out the right base_class for the right task
    if type(base_class) is dict:
        base_class = base_class[context["modality"][modality]["task"]]
    return base_class


def get_output_dir_from_context(context: Dict) -> Path:
    # we need to find out which base class to use, let's get the modality e.g. `vision`
    modality = list(context["modality"].keys())[0]
    # now from the base_classes dict, use the modality to get it's base_class
    model_type = context["modality"][modality]["type"] + "s"
    # if what we have found is a dict, it means that it can have tasks, we need to find out the right base_class for the right task
    task = ""
    if "task" in context["modality"][modality]:
        task = context["modality"][modality]["task"]
        # task is something like image-classification, our directory structure follows image/classification
        # model_type in this case is image
        model_type, task = task.split("-")
    output_dir = GLASSES_DIR / "models" / modality / model_type / task
    return output_dir


def add_new_model_command():

    context = {}
    questions = Questions(
        [
            Ask("model_name_lower_case", default="my_model"),
            Ask("ModelNameCammelCase", default="MyModel"),
            Ask("model_checkpoint", default="my_model_base"),
            Select(
                "modality",
                {
                    "vision": Questions(
                        [
                            Choices("type", ["model", "head", "backbone", "neck"]),
                            # here we should ask for the task only if you select model and head
                            IfHasSelected(
                                ["model", "head"],
                                Choices(
                                    "task",
                                    [
                                        "image-classification",
                                        "image-detection",
                                        "image-segmentation",
                                    ],
                                ),
                            ),
                        ]
                    )
                },
            ),
        ]
    )

    questions(context)

    print_json(data=context)

    confirm("Is this okay?")

    # update the context
    context["base_class"] = get_base_class_from_context(context)
    context["ConfigNameCammelCase"] = f"{context['ModelNameCammelCase']}Config"

    output_dir = get_output_dir_from_context(context)
    generate_files(
        TEMPLATE_DIR / "adding_new_model",
        context={"cookiecutter": context},
        overwrite_if_exists=True,
        output_dir=output_dir,
    )

    print(f"Your model has been created at {output_dir} 🎉")
