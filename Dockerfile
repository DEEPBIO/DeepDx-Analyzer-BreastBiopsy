FROM nvidia/cuda:10.2-cudnn8-runtime-ubuntu18.04

RUN apt-get -qq update

# Install python3.8 and pip
RUN apt-get -qq install -y --no-install-recommends python3.8 python3.8-dev python3.8-distutils
RUN apt-get -qq install -y --no-install-recommends python3-pip
RUN update-alternatives --install /usr/bin/python python $(which python3.8) 1
RUN update-alternatives --install /usr/bin/python3 python3 $(which python3.8) 1
RUN update-alternatives --install /usr/bin/pip pip $(which pip3) 1


# Install libraries for opencv.
RUN apt-get -qq install ffmpeg libsm6 libxext6  -y
# Install openslide.
RUN apt-get -qq install -y openslide-tools

# Install build tools for openslide-python
RUN pip install -U 'setuptools==45.2.0'
RUN apt-get -qq install -y build-essential
RUN pip install wheel

RUN apt-get install -qq wget
# Apply updated openslide binary (see https://github.com/DEEPBIO/deepbio-openslide)
RUN wget https://openslide-binary.s3.amazonaws.com/ubuntu_openslidetools/libopenslide.so.0 -O /usr/lib/x86_64-linux-gnu/libopenslide.so.0

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt && rm requirements.txt

WORKDIR /root
