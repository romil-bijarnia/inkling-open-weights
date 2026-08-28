FROM node:22-bookworm-slim AS web-build
WORKDIR /src/web
COPY web/package.json ./
RUN npm install
COPY web/ ./
RUN npm run build

FROM python:3.12-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models/huggingface \
    NEURAL_ATLAS_HOST=0.0.0.0 \
    NEURAL_ATLAS_PORT=8000
WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY neural_atlas/ ./neural_atlas/
COPY --from=web-build /src/web/dist ./web/dist/
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -e .

EXPOSE 8000
VOLUME ["/models/huggingface"]
CMD ["uvicorn", "neural_atlas.app:app", "--host", "0.0.0.0", "--port", "8000"]
