FROM ubuntu:18.04

# To make it easier for build and release pipelines to run apt-get,
# configure apt to not require confirmation (assume the -y argument by default)
ENV DEBIAN_FRONTEND=noninteractive
RUN echo "APT::Get::Assume-Yes \"true\";" > /etc/apt/apt.conf.d/90assumeyes

RUN apt-get update && apt-get install software-properties-common \
    && add-apt-repository ppa:git-core/ppa \
    && apt-get update \
    && apt-get install git
RUN apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    ca-certificates \
    curl \
    jq \
    git \
    iputils-ping \
    libcurl4 \
    libicu60 \
    libunwind8 \
    netcat \
    libssl1.0
RUN pip3 install --upgrade pip
RUN pip3 install category-encoders \
    gitpython \
    tqdm \
    prettytable \
    cryptography \
    typing-extensions

COPY ./src/sorts ./sorts
