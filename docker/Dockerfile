FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-runtime

RUN apt-get update        
RUN apt-get install -y git

COPY . /opt/app
WORKDIR /opt/app
RUN pip install -r requirements.txt
RUN pip install -U kaleido
WORKDIR /home
RUN pip install jupyter -U && pip install jupyterlab
