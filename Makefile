check_dirs := glasses

style:
	black --preview $(check_dirs)
	isort $(check_dirs)

check_code_quality:
	black --check --preview $(check_dirs)
	isort --check-only $(check_dirs)
	flake8 $(check_dirs)