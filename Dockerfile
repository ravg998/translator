FROM --platform=linux/amd64 python:3.12-slim 


ENV PYTHONBUFFERED=1 

# PACKAGES 
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \ 
    && rm -rf /var/lib/apt/lists/*


RUN pip install --no-cache-dir uv 


WORKDIR /app 

# PYTHON 
COPY uv.lock pyproject.toml ./
RUN uv sync --frozen --no-install-project


COPY . .
RUN uv sync --frozen 

CMD ["tail", "-f", "/dev/null"]


