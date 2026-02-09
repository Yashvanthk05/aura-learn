FROM python:3.10-slim

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN pip install uv && uv sync --no-dev

COPY . .

CMD ["uv","run","python","src/auralearn/main.py"]