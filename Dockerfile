FROM python:3.7-buster

RUN apt-get update && apt-get upgrade -y \
    && pip install pipenv
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

ARG user
RUN adduser --disabled-password --gecos '' ${user}
USER ${user}
RUN mkdir /home/${user}/dev
WORKDIR /home/${user}/dev

COPY jupyter_config.txt Pipfile ./
RUN pipenv install --dev && \
    pipenv run jupyter notebook --generate-config && \
    cat jupyter_config.txt >> ~/.jupyter/jupyter_notebook_config.py
