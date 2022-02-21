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
    python3.8 \
    python3.8-venv \
    python3-venv

RUN python3.8 -m venv /venv
ENV PATH /venv/bin:$PATH

COPY src/sorts/res/requirements.txt .
RUN pip3 install --upgrade pip \
    && pip3 install -r requirements.txt

COPY ./src/sorts ./sorts

RUN chmod +x ./sorts/entrypoint.py
RUN echo '/sorts/entrypoint.py "$@"' > /usr/bin/sorts \
    && chmod +x /usr/bin/sorts
