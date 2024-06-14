FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV PYTHON_VERSION=3.11.9

# (Optional) Change Ubuntu package repository to JAIST mirror site
RUN sed -i.bak -e "s%http://archive.ubuntu.com/ubuntu/%http://ftp.jaist.ac.jp/pub/Linux/ubuntu/%g" /etc/apt/sources.list

# Install necessary packages
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    build-essential \
    curl \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Pyenv
ENV HOME=/root
ENV PYTHON_ROOT=$HOME/local/python-$PYTHON_VERSION
ENV PATH=$PYTHON_ROOT/bin:$PATH
ENV PYENV_ROOT=$HOME/.pyenv
RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
    git \
    make \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    wget \
    curl \
    llvm \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libffi-dev \
    liblzma-dev \
    && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/pyenv/pyenv.git $PYENV_ROOT \
    && $PYENV_ROOT/plugins/python-build/install.sh \
    && /usr/local/bin/python-build -v $PYTHON_VERSION $PYTHON_ROOT \
    && rm -rf $PYENV_ROOT

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH=/root/.local/bin:$PATH
RUN poetry config virtualenvs.create false

# Install dependencies by Poetry
WORKDIR /workspace
COPY env/pyproject.toml env/poetry.lock ./
RUN poetry install --no-dev

CMD ["tail", "-f", "/dev/null"]