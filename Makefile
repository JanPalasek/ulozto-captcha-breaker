SHELL := /bin/bash
PYTHON ?= python

init:
	sh scripts/init.sh

clean:
	rm -rf .quarto _site/

install_dev:
	$(PYTHON) -m pip install -r requirements.txt -e .


.PHONY: init clean install_dev