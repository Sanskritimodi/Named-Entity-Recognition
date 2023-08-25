#FROM nvidia/cuda:11.4.3-devel-ubuntu20.04
FROM ubuntu:20.04
RUN apt-get update && apt-get install -y gcc musl-dev g++
RUN apt-get install -y python3 python3-dev python3-pip

COPY requirements.txt /

RUN pip3 install -r /requirements.txt

WORKDIR /root
COPY config.yaml .
COPY train.py .
COPY evaluate.py .

RUN mkdir -p data/
RUN mkdir -p models/
