FROM nvcr.io/nvidia/tritonserver:24.05-py3

ENV CUDA_HOME=/usr/local/cuda-12.4
ENV PATH=$CUDA_HOME/bin:$PATH

RUN pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 --extra-index-url https://download.pytorch.org/whl/cu124

WORKDIR /workspace
COPY --chown=triton-server:triton-server ./repository models

CMD ["tritonserver", "--model-repository=/workspace/models"]