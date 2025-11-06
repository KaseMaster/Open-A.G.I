FROM python:3.10-slim
WORKDIR /app
COPY . /app
COPY docker/requirements-docker.txt ./requirements-docker.txt
RUN pip install --no-cache-dir -r requirements-docker.txt
ENV NODE_ROLE=validator
ENV VALIDATOR_ID=1
EXPOSE 8000
CMD ["python", "scripts/validator_node.py"]