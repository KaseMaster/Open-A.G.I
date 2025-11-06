FROM python:3.10-slim
WORKDIR /app
COPY . /app
COPY docker/requirements-docker.txt ./requirements-docker.txt
RUN pip install --no-cache-dir -r requirements-docker.txt
ENV NODE_ROLE=client
EXPOSE 9000
CMD ["python", "scripts/client_node.py"]