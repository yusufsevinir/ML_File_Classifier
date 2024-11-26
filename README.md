# Heron File Classifier

## Overview

Welcome to the **Heron File Classifier** project! This application is designed to classify files based on their content rather than just their filenames. It supports various file types and uses both rule-based and machine learning classifiers to enhance accuracy. The classifier is structured to handle large volumes of files efficiently, making it scalable and suitable for production environments.

## Features

- **Content-Based Classification**: Extracts text from files and classifies them based on the actual content.
- **Support for Multiple File Types**: Handles PDFs, images (JPEG, PNG), Word documents, Excel files, plain text, and RTF files.
- **Machine Learning Classifier**: Incorporates a trained ML model alongside rule-based classification for improved accuracy.
- **Scalable Architecture**: Modular codebase designed for easy maintenance and scalability.
- **Synthetic Data Generation**: Generates synthetic data for training the ML model, facilitating expansion to new industries.
- **Comprehensive Logging and Debugging**: Detailed logging for easier troubleshooting and monitoring.
- **Rate Limiting**: Implements rate limiting to prevent abuse and ensure fair usage.
- **Extensible Design**: Easy to add support for new file types and document categories.

## Table of Contents

- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Environment Variables](#environment-variables)
- [Usage](#usage)
  - [Running the Application](#running-the-application)
  - [API Endpoints](#api-endpoints)
  - [Testing the Classifier](#testing-the-classifier)
- [Training the ML Model](#training-the-ml-model)
- [Project Components](#project-components)
  - [Application Entry Point (`app.py`)](#application-entry-point-apppy)
  - [Classifiers](#classifiers)
    - [Base Classifier (`base_classifier.py`)](#base-classifier-base_classifierpy)
    - [Rule-Based Classifier (`rule_based_classifier.py`)](#rule-based-classifier-rule_based_classifierpy)
    - [Machine Learning Classifier (`ml_classifier.py`)](#machine-learning-classifier-ml_classifierpy)
  - [Text Extraction (`text_extractor.py`)](#text-extraction-text_extractorpy)
  - [Synthetic Data Generation (`synthetic_data_generator.py`)](#synthetic-data-generation-synthetic_data_generatorpy)
  - [Model Training (`train_model.py`)](#model-training-train_modelpy)
  - [Utilities](#utilities)
    - [Text Preprocessing (`text_preprocessing.py`)](#text-preprocessing-text_preprocessingpy)
    - [File Utilities (`file_utils.py`)](#file-utilities-file_utilspy)
- [Running Tests](#running-tests)
- [Future Improvements](#future-improvements)
- [License](#license)

## Getting Started

### Prerequisites

- **Python 3.9+**
- **Tesseract OCR**: Required for text extraction from images.
- **System Dependencies**: Libraries for processing various file types.

#### Install System Dependencies

**Ubuntu/Debian:**

```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr libtesseract-dev libleptonica-dev \
                        pkg-config poppler-utils libmagic1 \
                        libpoppler-cpp-dev python3-dev

macOS (using Homebrew):

brew install tesseract libmagic poppler

Installation

	1.	Clone the Repository:

git clone <repository_url>
cd join-the-siege


	2.	Set Up Virtual Environment and Install Dependencies:

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt



Environment Variables

Create a .env file in the project root to configure environment variables:

FLASK_APP=src.app
FLASK_ENV=development
CLASSIFIER_TYPE=ml_based  # Options: 'rule_based', 'ml_based'
LOG_LEVEL=INFO

Usage

Running the Application

	1.	Activate the Virtual Environment:

source venv/bin/activate


	2.	Run the Flask App:

flask run

The application will start on http://127.0.0.1:5000/.

API Endpoints

	•	GET /health
Health check endpoint.
Response:

{
  "status": "healthy"
}


	•	POST /classify_file
Upload a file to classify.
Parameters:
	•	file: The file to classify.
Response:
	•	200 OK: Returns the classification result and processing time.

{
  "file_class": "invoice",
  "processing_time": 0.234
}


	•	400 Bad Request: If the file is missing or invalid.
	•	429 Too Many Requests: If rate limit is exceeded.
	•	500 Internal Server Error: If an unexpected error occurs.

Testing the Classifier

Submit a File for Classification:

curl -X POST -F 'file=@path_to_file' http://127.0.0.1:5000/classify_file

Check Health Status:

curl http://127.0.0.1:5000/health

Training the ML Model

Before running the application with the ML classifier, you need to train the machine learning model.
	1.	Generate Synthetic Data and Train the Model:

python src/training/train_model.py

This script generates synthetic data using predefined templates and trains a BERT-based classifier.

	2.	Model and Tokenizer Files:
The trained model and tokenizer are saved in the src/models/document_classifier directory.

Project Components

Application Entry Point (app.py)

Located at src/app.py, this file initializes the Flask application, configures logging, sets up rate limiting, and defines the API endpoints.

Key Features:
	•	Rate Limiting: Prevents abuse by limiting the number of requests per user.
	•	Logging: Provides detailed logs for debugging and monitoring.
	•	Health Check Endpoint: Allows monitoring tools to verify the application’s health.
	•	TODOs for Future Improvements: Includes placeholders for enhancements like authentication, async processing, and caching.

Notes:
	•	The application uses the Flask-Limiter library for rate limiting.
	•	Logging is configured based on environment variables for flexibility.
	•	The classifier is initialized based on the CLASSIFIER_TYPE environment variable, allowing easy switching between rule-based and ML classifiers.

Classifiers

Base Classifier (base_classifier.py)

An abstract base class that defines the interface for classifiers.

Key Features:
	•	Abstract Method: Defines the classify method that must be implemented by subclasses.
	•	Standardization: Ensures all classifiers adhere to the same interface.

Rule-Based Classifier (rule_based_classifier.py)

Implements a simple rule-based approach using keyword matching.

Key Features:
	•	Keyword Matching: Searches for specific keywords in the extracted text to determine the document type.
	•	Logging: Provides debug information about the classification process, aiding in troubleshooting.
	•	Quick Initialization: Does not require extensive setup or training, making it lightweight.

Machine Learning Classifier (ml_classifier.py)

Uses a pre-trained Transformer model (BERT) for classification.

Key Features:
	•	Transformer Model: Utilizes a fine-tuned BERT model for better accuracy in classification tasks.
	•	Confidence Scores: Provides confidence levels for predictions, allowing for more informed decisions.
	•	Debug Information: Includes detailed logging and debug data for each classification, helpful for monitoring and improving the model.
	•	GPU Support: Automatically uses GPU if available, enhancing performance for large-scale deployments.

Notes:
	•	The classifier loads the model and tokenizer from the models/document_classifier directory.
	•	Input texts are preprocessed and tokenized before being fed into the model.
	•	Predictions include top labels with their corresponding confidence scores.

Text Extraction (text_extractor.py)

Located at src/extraction/text_extractor.py, this module handles extracting text from various file types.

Supported File Types:
	•	PDF: Extracts text from PDF documents using pypdf.
	•	Images (JPEG, PNG): Uses Tesseract OCR to extract text from images.
	•	Word Documents: Extracts text from .docx files using docx2txt.
	•	Excel Spreadsheets: Reads data from .xlsx files using openpyxl.
	•	RTF Files: Converts RTF files to plain text using striprtf.
	•	Plain Text and CSV Files: Reads and decodes text files directly.

Key Features:
	•	MIME Type Detection: Determines the file type based on MIME type for accurate processing.
	•	Error Handling: Includes robust error handling and logging for troubleshooting extraction issues.
	•	Logging: Provides detailed logs at each step of the extraction process.

Synthetic Data Generation (synthetic_data_generator.py)

Generates synthetic documents for training the ML model.

Key Features:
	•	Randomized Data: Uses the faker library to generate realistic and diverse data points.
	•	Templates: Loads templates from JSON files to structure the synthetic data, ensuring consistency.
	•	Multiple Document Types: Generates synthetic data for bank statements, invoices, and driver’s licenses.
	•	Metadata Inclusion: Stores metadata alongside text content for potential use in advanced training scenarios.

Notes:
	•	The module can generate a customizable number of samples.
	•	Synthetic data helps in training the ML model when real data is scarce or sensitive.

Model Training (train_model.py)

Trains a BERT-based model using the synthetic dataset.

Key Features:
	•	Data Preparation: Splits data into training and validation sets, ensuring a fair evaluation.
	•	Model Configuration: Sets up training arguments optimized for document classification tasks.
	•	Metrics Calculation: Computes accuracy, precision, recall, and F1 score for comprehensive evaluation.
	•	Early Stopping: Uses callbacks to prevent overfitting by stopping training when no improvement is observed.
	•	Model Saving: Saves the trained model, tokenizer, and label mappings for future use.

Notes:
	•	The script increases the dataset size and adjusts training parameters for better performance.
	•	Utilizes the transformers library for model training and management.
	•	The trained model is stored in a designated directory for consistency.

Utilities

Text Preprocessing (text_preprocessing.py)

Provides functions for cleaning and preparing text data.

Key Features:
	•	Lowercasing: Converts all text to lowercase for uniformity.
	•	Special Character Removal: Removes URLs, non-alphabetic characters, and extraneous symbols.
	•	Whitespace Normalization: Cleans up extra spaces to streamline the text.
	•	Alphabetic Filtering: Extracts alphabetic characters from alphanumeric strings for focus on meaningful content.

File Utilities (file_utils.py)

Contains utility functions for file handling, including MIME type validation.

Key Features:
	•	MIME Type Checking: Validates if a file is of an allowed type using the python-magic library.
	•	Supported MIME Types: Defined in ALLOWED_MIME_TYPES for easy modification and extension.
	•	File Rewinding: Ensures the file pointer is reset after reading for consistent processing.

Notes:
	•	Centralizes file validation logic, making it easier to maintain and update supported file types.
	•	Enhances security by preventing processing of disallowed or potentially harmful file types.

Running Tests

Tests are located in the tests directory and cover various components of the application.

Running All Tests:

pytest

Test Coverage:
	•	Classifiers: Ensures both the rule-based and ML classifiers function as expected.
	•	Text Extraction: Verifies that text is correctly extracted from supported file types.
	•	Utilities: Tests utility functions like text preprocessing and file validation.
	•	Application Routes: Checks the API endpoints for correct responses and error handling.

Future Improvements

	•	Asynchronous Processing: Implement Celery or another task queue for handling large files asynchronously. Or fast api for async processing.
	•	Caching: Add caching mechanisms to store results of frequently classified documents, improving performance.
	•	Authentication: Implement API key authentication and role-based access control for enhanced security.
	•	Monitoring and Metrics: Integrate tools like Prometheus and Grafana for monitoring performance and resource usage.
	•	Error Handling Enhancements: Add more granular error handling, retry mechanisms, and user-friendly error messages.
	•	Scalability: Deploy the application using a production-grade WSGI server like Gunicorn and consider containerization with Docker or Kubernetes.



