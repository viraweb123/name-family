FROM 139.99.239.192:5000/nvcr.io/nvidia/pytorch:23.12-py3

ENV TRANSFORMERS_CACHE=/train/.cache/huggingface

RUN pip3 install --no-cache-dir transformers[torch]

WORKDIR /train/code/


COPY ../code/ /train/code/
COPY ../input/ /train/input/
COPY ../log/ /train/log/


RUN mkdir -p /train/.cache/huggingface


RUN mkdir -p /train/output && chmod -R 777 /train/output /train/log


CMD [ "python3", "main.py" ]