FROM ubuntu:20.04 AS builder

ENV DEBIAN_FRONTEND="noninteractive" TZ="Europe/Amsterdam"

# Setup
RUN apt-get update && apt-get install --no-install-recommends -y \
     build-essential \
     cmake-gui \
     mesa-utils \
     xorg-dev \
     libglu1-mesa-dev \
     libboost-all-dev \
    && rm -rf /var/lib/apt/lists/*

# BUILD AdTree
RUN mkdir /usr/src/AdTree && cd /usr/src/AdTree
WORKDIR /usr/src/AdTree
COPY AdTree /usr/src/AdTree

# RUN mkdir Release && cd Release
WORKDIR Release
RUN cmake -DCMAKE_BUILD_TYPE=Release ..
RUN make

# SECOND_STAGE
FROM ubuntu:20.04 AS runtime

ENV DEBIAN_FRONTEND="noninteractive" TZ="Europe/Amsterdam"

COPY --from=builder /usr/src/AdTree/Release/bin/AdTree /usr/local/app/bin/AdTree

RUN apt-get update && apt-get install --no-install-recommends -y \
     libgl1 \
     libgomp1 \
     libglu1-mesa-dev \
     python3 \
     python3-pip \
     && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/local/app

# PYTHON
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir --upgrade jupyter 
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt && rm requirements.txt

# JUPYTER MODE
ENTRYPOINT ["jupyter", "notebook", "--ip=0.0.0.0", "--no-browser", "--allow-root"]

# BUILD COMMAND
# docker build -f Dockerfile . -t tree-modelling:latest

# RUN COMMAND
# docker run -v `pwd`/pctm:/usr/local/app/pctm -v `pwd`/dataset:/usr/local/app/dataset -it -p 8888:8888 tree-modelling:latest
# docker run -v `pwd`/pctm:/usr/local/app/pctm -v `pwd`/dataset:/usr/local/app/dataset -it --entrypoint /bin/bash tree-modelling:latest

