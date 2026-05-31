FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential git && rm -rf /var/lib/apt/lists/*
COPY requirements.txt pyproject.toml README.md /app/
COPY src /app/src
COPY tests /app/tests
COPY examples /app/examples
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir -e .
CMD ["sh", "-c", "python -m pytest && python examples/minimal_fpde_example.py"]
