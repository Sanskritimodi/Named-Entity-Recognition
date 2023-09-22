#FROM nvidia/cuda:11.4.3-devel-ubuntu20.04
FROM ubuntu:20.04
RUN apt-get update && apt-get install -y gcc musl-dev g++
RUN apt-get install -y python3 python3-dev python3-pip wget gzip

COPY requirements.txt /

RUN pip3 install -r /requirements.txt

WORKDIR /root
COPY setup.sh .
COPY data/ /root/data/

#RUN wget -c https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.gu.300.vec.gz -O /root/ft/cc.gu.300.vec.gz
#RUN gunzip root/ft/cc.gu.300.vec.gz 

RUN mkdir -p ft/
RUN wget -c https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.gu.300.vec.gz -O ft/cc.gu.300.vec.gz
RUN gunzip ft/cc.gu.300.vec.gz 

COPY config.yaml .
COPY bio_to_json.py .
COPY train.sh .
COPY evaluate.py .

RUN mkdir -p data/
RUN mkdir -p models/

RUN chmod +x setup.sh
#RUN python3 bio_to_json.py
RUN python3 -m stanza.models.common.convert_pretrain ~/stanza_resources/gu/pretrain/fasttext.pt ~/ft/cc.gu.300.vec 150000 
RUN chmod +x train.sh
