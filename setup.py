from setuptools import find_packages, setup

with open("./README.md", "r") as f:
    long_description = f.read()

with open("./requirements.txt", "r") as f:
    install_requires = f.read().split("\n")

setup(
    name="glasses",
    version="0.1.0",
    author="Francesco Saverio Zuppichini & Anugunj Naman",
    author_email="francesco.zuppichini@gmail.com",
    description="Compact, concise and customizable deep learning computer vision.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/FrancescoSaverioZuppichini/glasses",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=install_requires,
    python_requires=">=3.9",
    extras_require={
        "dev": ["pytest", "flake8", "cookiecutter==2.1.1", "black==22.3.0"],
        "doc": [
            "markdown",
            "mkdocs-autorefs",
            "mkdocs-material",
            "pymdown-extensions",
            "mkdocstrings",
            "mkdocs-gen-files",
            "mkdocs-literate-nav",
            "mkdocstrings-python",
        ],
    },
    entry_points={
        "console_scripts": [
            "glasses-cli = cli.main:app",
        ],
    },
)
