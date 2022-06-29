from cli.questions import Questions, Ask, Select, IfHasSelected, Choices
from rich import print_json
from cookiecutter.generate import generate_files


def add_new_model_command():

    base_classes = {
        "vision": {
            "neck": "Neck",
            "backbone": "Backbone",
            "model": {"image-classification": "ModelForClassification"},
            "head": {"image-classification": "HeadForClassification"},
        }
    }
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

    modality = list(context["modality"].keys())[0]

    base_class = base_classes[modality][context["modality"][modality]["type"]]

    if type(base_class) is dict:
        base_class = base_class[context["modality"][modality]["task"]]

    confirmation = input("Is this okay? [y/n]: ")
    if confirmation == "n":
        return

    context["base_class"] = base_class
    context["ConfigNameCammelCase"] = f"{context['ModelNameCammelCase']}Config"

    generate_files(
        "/home/zuppif/Documents/glasses-2.0/templates/adding_new_model/",
        context={"cookiecutter": context},
        overwrite_if_exists=True,
        output_dir="/home/zuppif/Documents/glasses-2.0/template_out",
    )


add_new_model_command()
