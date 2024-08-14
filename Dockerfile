FROM 139.99.239.192:5000/nvcr.io/nvidia/pytorch:23.12-py3

ENV HF_HOME=/train/.cache/huggingface

RUN pip3 install --no-cache-dir transformers[torch]

WORKDIR /train/code/


COPY ./train/code/ /train/code/
COPY ./train/input/ /train/input/
COPY ./train/log/ /train/log/


RUN mkdir -p /train/.cache/huggingface


RUN mkdir -p /train/output && chmod -R 777 /train/output /train/log


ENTRYPOINT ["python3", "main.py"] > /train/log/docker.log 2>&1