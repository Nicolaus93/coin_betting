.DEFAULT_GOAL := help
.PHONY: coverage deps help lint push test tox

coverage:  ## Run tests with coverage
    coverage erase
    coverage run --include=optimal_pytorch/* -m pytest tests/
    coverage report -m

deps:  ## Install dependencies
    pip install black coverage flake8 flit mccabe mypy pylint pytest tox tox-gh-actions

lint:  ## Lint and static-check
    flake8 optimal_pytorch/ examples/MNIST examples/separable_data.py --max-line-length 88
    pylint optimal_pytorch/coin_betting/
    mypy optimal_pytorch/

push:  ## Push code with tags
    git push && git push --tags

black:
    black optimal_pytorch/ examples/

tox:   ## Run tox
    python -m tox

help: ## Show help message
    @IFS=$$'\n' ; \
    help_lines=(`fgrep -h "##" $(MAKEFILE_LIST) | fgrep -v fgrep | sed -e 's/\\$$//' | sed -e 's/##/:/'`); \
    printf "%s\n\n" "Usage: make [task]"; \
    printf "%-20s %s\n" "task" "help" ; \
    printf "%-20s %s\n" "------" "----" ; \
    for help_line in $${help_lines[@]}; do \
        IFS=$$':' ; \
        help_split=($$help_line) ; \
        help_command=`echo $${help_split[0]} | sed -e 's/^ *//' -e 's/ *$$//'` ; \
        help_info=`echo $${help_split[2]} | sed -e 's/^ *//' -e 's/ *$$//'` ; \
        printf '\033[36m'; \
        printf "%-20s %s" $$help_command ; \
        printf '\033[0m'; \
        printf "%s\n" $$help_info; \
    done