
FROM python:3.10-slim

# Install all dependencies at build time — never at runtime
RUN pip install mlflow==2.14.0 psycopg2-binary boto3 --no-cache-dir

# Expose port
EXPOSE 5000
# No pip install at startup — just run the server
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]
