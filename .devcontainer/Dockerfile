FROM python:3.11-slim-bookworm
USER root

RUN apt update && \
    apt -y install --reinstall ca-certificates && \
    apt -y install software-properties-common

COPY requirements.txt ./
RUN pip --no-cache-dir install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt
