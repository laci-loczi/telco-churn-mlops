# Pinned MLflow tracking server image.
#   psycopg2-binary: Postgres driver for the backend store
#   boto3:           S3 SDK for the artifact store (Silo)
#   sqlalchemy<2.1:  2.1 switched the default postgresql:// driver to psycopg 3
FROM python:3.12-slim

RUN pip install --no-cache-dir \
    mlflow==3.13.0 \
    psycopg2-binary==2.9.12 \
    boto3==1.43.29 \
    "sqlalchemy<2.1"

EXPOSE 5000

ENTRYPOINT ["mlflow", "server"]
