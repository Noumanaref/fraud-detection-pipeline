# Real-Time Fraud Detection Data Engineering Pipeline

[![CI/CD Pipeline](https://github.com/[YOUR_GITHUB_USERNAME]/fraud-detection-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/[YOUR_GITHUB_USERNAME]/fraud-detection-pipeline/actions)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![Apache Spark](https://img.shields.io/badge/Spark-3.5.0-orange.svg)](https://spark.apache.org/)
[![Delta Lake](https://img.shields.io/badge/Delta_Lake-3.1.0-lightblue.svg)](https://delta.io/)
[![MLflow](https://img.shields.io/badge/MLflow-Registry-green.svg)](https://mlflow.org/)

An end-to-end data engineering and MLOps system that processes **3.5 million+ financial transaction records**, trains machine learning models to detect fraud, and serves live predictions for business monitoring and alerting[cite: 1, 2].

---

## 🏗️ System Architecture

The project follows a modern **Medallion Lakehouse** architecture combined with a **Kappa streaming design**, ensuring raw data is safely stored while fast analytics run smoothly.

Read the full [Comprehensive Technical Documentation](docs/technical_documentation.md) for a deep dive into the architecture and optimizations.
