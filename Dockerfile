#FROM  huggingface/transformers-pytorch-gpu:4.41.2
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel


RUN apt-get update && apt-get install -y vim

RUN pip --version
RUN python --version

COPY requirements.txt /tmp/
RUN pip install Cython fastapi==0.110 
#RUN pip install spacy==3.0.6
RUN pip install spacy
RUN pip install -r /tmp/requirements.txt

