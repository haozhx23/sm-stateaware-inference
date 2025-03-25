
# FROM python:3.10
# FROM nvcr.io/nvidia/pytorch:23.07-py3 # cu12.1.1

FROM lmsysorg/sglang:latest

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE

ENV HF_TOKEN=hf_

LABEL version="0.1"
LABEL description="sglang serving"

EXPOSE 8080

RUN mkdir -p /workspace
WORKDIR /workspace

# COPY sglang/ /workspace/sglang/

COPY requirements.txt /workspace/requirements.txt
COPY main.py /workspace/main.py
COPY s5cmd /workspace/s5cmd
RUN chmod +x s5cmd

RUN pip install --upgrade pip
RUN pip install "sglang[all]"
RUN pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/
RUN pip install -r requirements.txt

ENTRYPOINT ["python3", "/workspace/main.py"]
# ENTRYPOINT ["python", "-m", "sglang.launch_server", "--model-path", "meta-llama/Llama-3.1-8B-Instruct", "--host", "0.0.0.0", "--port", "8080"]

