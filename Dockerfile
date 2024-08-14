FROM 139.99.239.192:5000/nvcr.io/nvidia/pytorch:23.12-py3

RUN pip3 install --no-cache-dir transformers[torch]
