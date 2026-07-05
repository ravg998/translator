FROM python:3.12-slim 

ARG HUG_USER="hugfacegva98/translator"
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
RUN uv run hf download ${HUG_USER} --local-dir /app/data

CMD ["tail", "-f", "/dev/null"]