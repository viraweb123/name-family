FROM nvcr.io/nvidia/pytorch:22.01-py3

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
        git \
        python3-pip \
        python3-dev \
        python3-opencv \
        libglib2.0-0
# Install any python packages you need
# COPY requirements.txt requirements.txt

# RUN python3 -m pip install -r requirements.txt

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install PyTorch and torchvision
RUN pip3 install transformers

# Set the working directory
WORKDIR /app

COPY . .

# Set the entrypoint
ENTRYPOINT [ "python3" ]

CMD [ "main.py" ]