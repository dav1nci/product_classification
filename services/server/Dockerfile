FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel

RUN apt-get update && apt-get install -y vim

COPY requirements.txt /tmp/
RUN pip install Cython fastapi==0.110
RUN pip install -r /tmp/requirements.txt
