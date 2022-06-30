.PHONY: style check_code_quality

export PYTHONPATH = .
check_dirs := glasses

style:
	black  $(check_dirs)
	isort --profile black $(check_dirs)

check_code_quality:
	black --check $(check_dirs)
	isort --check-only --profile black $(check_dirs)
	# stop the build if there are Python syntax errors or undefined names
	flake8 $(check_dirs)--count --select=E9,F63,F7,F82 --show-source --statistics
	# exit-zero treats all errors as warnings. E203 for black, E501 for docstring
	flake8 $(check_dirs) --count --max-line-length=88  --ignore=D--extend-ignore=E203,E501  --statistics
