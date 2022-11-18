ARG VARIANT="3.10-bullseye"
FROM mcr.microsoft.com/vscode/devcontainers/python:0-${VARIANT} AS devcontainer
WORKDIR /workspaces/badger_drill_cv

ENV PATH="${PATH}:/home/vscode/.local/bin:/root/.local/bin"
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y

USER vscode
COPY requirements.txt .
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

ARG NODE_VERSION="none"
USER vscode
RUN if [ "${NODE_VERSION}" != "none" ]; then umask 0002 && . /usr/local/share/nvm/nvm.sh && nvm install ${NODE_VERSION} 2>&1; fi

# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.10-slim AS deploy

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

# Install production dependencies.
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install --no-cache-dir -r requirements.txt

CMD gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app
