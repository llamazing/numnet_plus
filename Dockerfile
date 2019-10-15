FROM python:3.6-stretch

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Install base packages.
RUN apt-get update --fix-missing && apt-get install -y \
    bzip2 \
    ca-certificates \
    curl \
    gcc \
    git \
    libc-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    wget \
    libevent-dev \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /local

RUN pip install -U scikit-learn

# Install dependencies.
COPY ./requirements.txt .
RUN pip install -r requirements.txt

# Copy remaining code.
COPY . .

RUN ["chmod", "+x", "/local/run_model.sh"]

# EXPOSE 8000
Cmd ["/bin/bash"]
