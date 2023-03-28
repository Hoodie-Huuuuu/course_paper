FROM ubuntu:jammy

ARG BASE_PACKAGES="python3 python3-pip protobuf-compiler"
ARG PYTHON_MODULES="grpcio grpcio-tools pyjwt protobuf"

ENV SEGMENTER_PORT=5001

RUN apt clean && \
    apt update && \
    apt install ${BASE_PACKAGES} -y

RUN pip3 install ${PYTHON_MODULES}

WORKDIR /app

COPY . .

RUN pip3 install -r ./requirements.txt

# generate grpc files
RUN python3 \
    -m grpc_tools.protoc \
    -I ./protos \
    --python_out=./ \
    --grpc_python_out=./ \
    ./protos/segmenter_service.proto

CMD ["python3", "-p=${SEGMENTER_PORT}", "segmenter_service.py"]