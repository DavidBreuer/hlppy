.PHONY: install docker docs tests baseplots all

# install package requirements and development version
install:
	pip install -r requirements.txt -r requirements_dev.txt \
	&& pip install --no-deps -e .

# build docker image and enable matplotlib support
# https://stackoverflow.com/a/46018699/5350621
# https://stackoverflow.com/a/18137056/5350621
docker:
	docker build -t hlppy . -f Dockerfile \
	&& docker run --rm -it \
   	--user=$(shell id -u) \
   	--env="DISPLAY" \
   	--workdir=/app \
   	--volume="$(shell pwd)":/app \
   	--volume="/etc/group:/etc/group:ro" \
   	--volume="/etc/passwd:/etc/passwd:ro" \
   	--volume="/etc/shadow:/etc/shadow:ro" \
   	--volume="/etc/sudoers.d:/etc/sudoers.d:ro" \
   	--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
   	hlppy bash

# create docs
# if the docs do not update properly,
# remove docs/build and docs/auto_examples and run:
# sphinx-apidoc -o docs hlppy
# cd docs
# make html
docs:
	cd docs \
	&& make html \
	&& (ln -s build/html/index.html index.html || true)

# run unit tests, doctests, linters, mpl tests, and create coverage
# --pylint-error-types=EF raises exceptions only for errors and failures
tests:
	NUMBA_DISABLE_JIT='1' \
	find . -name \*.pyc -delete && find . -name __pycache__ -exec rm -rv {} + \
	&& python -m pytest tests/ \
	--doctest-modules hlppy/ \
	--pylint hlppy/ tests/ --pylint-error-types=EF \
	--flakes hlppy/ tests/ \
	--mpl \
	--cov=hlppy --cov-report term --cov-report html:docs/build/html/coverage

# create matplotlib baseline plots as comparison files for mpl tests
baseplots:
	python -m pytest tests/plot --mpl-generate-path=tests/plot/baseline

# create docs, run tests, and open documentation in browser
all:
	(make docs ||true) \
	&& (make tests || true) \
	&& firefox docs/index.html
