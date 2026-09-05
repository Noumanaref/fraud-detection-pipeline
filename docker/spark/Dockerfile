FROM apache/spark:3.5.0

USER root

RUN pip install \
    delta-spark==3.1.0 \
    mlflow==2.14.0 \
    xgboost==2.0.0 \
    scikit-learn==1.3.0 \
    boto3==1.34.0 \
    psycopg2-binary==2.9.9 \
    requests==2.31.0 \
    importlib_metadata \
    --no-cache-dir

RUN mkdir -p /home/spark/.ivy2/cache
