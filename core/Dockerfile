FROM 139.99.239.192:5000/nvcr.io/nvidia/pytorch:23.12-py3

# # Set environment variables
# ENV DEBIAN_FRONTEND=noninteractive

# # Install system dependencies
# RUN apt-get update && \
#     apt-get install -y \
#         git \
#         python3-pip \
#         python3-dev \
#         python3-opencv \
#         libglib2.0-0
# # Install any python packages you need
# # COPY requirements.txt requirements.txt

# # RUN python3 -m pip install -r requirements.txt

# # Upgrade pip
# RUN python3 -m pip install --upgrade pip

# # Install PyTorch and torchvision
RUN pip3 install transformers

# # Set the working directory
WORKDIR /train

# COPY . .

# Set the entrypoint
ENTRYPOINT [ "python3" ]

CMD [ "code/main.py" ]