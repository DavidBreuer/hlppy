FROM python:3.7-stretch
LABEL maintainer="info <info@info.com>"

ARG DEPS=" \
    g++ \
    make \
    python3-dev"

WORKDIR /app
RUN apt-get update
RUN apt-get install -y --no-install-recommends $DEPS

ADD ./requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

ADD ./requirements_dev.txt /app/requirements_dev.txt
RUN pip install -r requirements_dev.txt

ADD . /app
RUN pip install --no-deps .
