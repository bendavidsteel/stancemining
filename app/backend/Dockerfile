FROM python:3.11 AS build

RUN apt-get update && \
    apt-get -y install gcc mono-mcs build-essential git libopenblas-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS" pip install --no-cache-dir --upgrade --user -r /code/requirements.txt && \
    pip cache purge

FROM python:3.11-slim

# Install OpenBLAS runtime libraries in the final image
RUN apt-get update && \
    apt-get -y install libopenblas0 libgomp1 && \
    rm -rf /var/lib/apt/lists/*

COPY --from=build /root/.local /root/.local

WORKDIR /code

COPY ./main.py /code/app/main.py

ENV PATH=/root/.local/bin:$PATH

CMD ["fastapi", "run", "app/main.py", "--port", "5000"]