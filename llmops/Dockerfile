FROM ghcr.io/mlflow/mlflow:v2.21.2

WORKDIR /app

COPY . .

RUN cd llama-cpp/gguf-py && \
    pip install --editable . && \
    cd ../..

RUN pip install -r requirements.txt

RUN cd LLaMA-Factory && \
    pip install -e ".[torch,metrics]"