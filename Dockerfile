# Change CUDA and cuDNN version here
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        wget \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-dev \
        python3.10-venv \
    && wget https://bootstrap.pypa.io/get-pip.py -O get-pip.py \
    && python3.10 get-pip.py \
    && rm get-pip.py \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/local/bin/pip3.10 /usr/local/bin/pip \
    && python --version \
    && pip --version \
    && apt-get purge -y --auto-remove software-properties-common \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

####### Add your own installation commands here #######
# RUN pip install some-package
# RUN wget https://path/to/some/data/or/weights
# RUN apt-get update && apt-get install -y <package-name>

WORKDIR /app
# Copy LICENSE, README and requirements.txt
COPY requirements.txt /app/requirements.txt
COPY LICENSE /app/LICENSE
COPY README.md /app/README.md

# Install litserve and requirements
RUN pip install --no-cache-dir litserve==0.2.3 -r requirements.txt

# Now copy python files and config file, not needed while installation but needed while running
# so we have a small layer and build time is faster when onyl changing code
COPY *.py /app/
COPY model_config.json /app/model_config.json

# Make port 8000 available to the world outside this container
EXPOSE 8000
CMD ["python", "/app/litfusion_api.py"]
