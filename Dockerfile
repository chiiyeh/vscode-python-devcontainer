FROM python:3.9-buster

RUN apt-get update && apt-get upgrade -y \
    && pip install pipenv
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

ARG user
RUN adduser --disabled-password --gecos '' ${user}
USER ${user}
RUN mkdir /home/${user}/dev
WORKDIR /home/${user}/dev

COPY Pipfile ./
RUN pipenv install --dev --skip-lock
COPY jupytext.toml /home/${user}/
