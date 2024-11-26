from flask import Flask, request, jsonify
from src.classifiers import get_classifier
from src.utils.file_utils import allowed_file
import logging
from werkzeug.middleware.proxy_fix import ProxyFix
import os
import time
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from redis.exceptions import RedisError
# TODO: Further improvements
# from celery import Celery  # For async task processing
# from flask_caching import Cache  # For response caching
# from prometheus_client import Counter, Histogram  # For metrics
# from opentelemetry import trace  # For distributed tracing

app = Flask(__name__)

# Proxy support for running behind nginx/other reverse proxies
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)

# Logging 
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Classifier initialization into a function to allow lazy loading and easier testing
def init_classifier():
    classifier_type = os.getenv('CLASSIFIER_TYPE', 'rule_based')
    return get_classifier(classifier_type=classifier_type)

classifier = init_classifier()

# Initialize rate limiter with Redis
try:
    limiter = Limiter(
        app=app,
        key_func=get_remote_address,
        default_limits=["200 per day", "50 per hour"],
        storage_uri=os.getenv("RATE_LIMIT_STORAGE_URL", "memory://")
    )
except RedisError as e:
    logging.error(f"Failed to connect to Redis: {e}")
    # Fallback to memory storage
    limiter = Limiter(
        app=app,
        key_func=get_remote_address,
        default_limits=["200 per day", "50 per hour"],
        storage_uri="memory://"
    )

"""
TODO: Performance & Scalability Improvements:
1. Add Redis caching for frequently classified document types
2. Implement batch processing endpoint for multiple files
3. Add async processing using Celery for large files
4. Implement response compression for large payloads
5. Add connection pooling for external services
"""

"""
TODO: Monitoring & Observability:
1. Add Prometheus metrics for:
   - Classification latency
   - Success/failure rates
   - Queue lengths
2. Implement OpenTelemetry tracing
3. Add health check metrics for dependencies
"""

# Example metrics (when prometheus_client is added)
# CLASSIFICATION_REQUESTS = Counter('classification_requests_total', 'Total classification requests')
# CLASSIFICATION_LATENCY = Histogram('classification_latency_seconds', 'Time spent processing classification')

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring."""
    return jsonify({"status": "healthy"}), 200

@app.route('/classify_file', methods=['POST'])
@limiter.limit("10 per minute")
def classify_file_route():
    """
    TODO: Improvements needed:
    1. Add file size validation
    2. Implement chunked upload for large files
    3. Add support for batch processing
    4. Implement caching for identical files
    5. Add async processing option with status endpoint
    """
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part in the request"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        if not allowed_file(file):
            return jsonify({"error": "File type not allowed"}), 400

        # Add request logging
        logging.info(f"Processing file: {file.filename}")
        
        # Add basic metrics
        start_time = time.time()
        file_class = classifier.classify(file)
        processing_time = time.time() - start_time
        logging.info(f"Classification completed in {processing_time:.2f}s")

        return jsonify({
            "file_class": file_class,
            "processing_time": processing_time
        }), 200
    except Exception as e:
        # Add more detailed error reporting
        logging.exception(f"Error processing file {file.filename if 'file' in locals() else 'unknown'}")
        return jsonify({
            "error": "Internal server error",
            "error_details": str(e) if app.debug else None
        }), 500

# Add rate limit exceeded handler
@app.errorhandler(429)  # Too Many Requests
def ratelimit_handler(e):
    return jsonify({
        "error": "Rate limit exceeded",
        "description": str(e.description)
    }), 429

"""
TODO: Additional Endpoints Needed:
1. Batch classification endpoint:
   @app.route('/classify_batch', methods=['POST'])
   - Accept multiple files
   - Return job ID for status tracking
   - Process asynchronously

2. Status checking endpoint:
   @app.route('/status/<job_id>', methods=['GET'])
   - Return job status and results
   - Include progress for batch jobs

3. Model information endpoint:
   @app.route('/model/info', methods=['GET'])
   - Return model version
   - List supported file types
   - Show current model performance metrics

4. Async classification endpoint:
   @app.route('/classify_async', methods=['POST'])
   - Accept large files
   - Return job ID immediately
   - Process in background
"""

"""
TODO: Authentication & Authorization:
1. Add API key authentication:
   @auth.require_api_key
   def protected_route():
       pass

2. Implement role-based access:
   - Admin roles for metrics/monitoring
   - User roles for basic classification
   - Rate limits based on user tier
"""

"""
TODO: Error Handling Improvements:
1. Add specific error types and handlers
2. Implement retry mechanism for transient failures
3. Add circuit breaker for external services
4. Implement graceful degradation
"""

# Example async implementation with Celery
"""
celery = Celery(app.name, broker=os.getenv('CELERY_BROKER_URL'))

@celery.task
def async_classify_file(file_data):
    try:
        result = classifier.classify(file_data)
        # Store result in Redis/Database
        return result
    except Exception as e:
        # Handle errors, retry if needed
        raise
"""

# Example cache implementation
"""
cache = Cache(config={'CACHE_TYPE': 'redis'})
cache.init_app(app)

@cache.memoize(timeout=3600)
def get_cached_classification(file_hash):
    # Return cached classification if available
    pass
"""

if __name__ == '__main__':
    """
    TODO: Production Deployment:
    1. Use production-grade WSGI server (Gunicorn)
    2. Implement proper signal handling
    3. Add graceful shutdown
    4. Configure proper logging handlers
    5. Set up metrics export
    """
    # Don't run in debug mode in production
    debug_mode = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(
        host='0.0.0.0',
        port=int(os.getenv('PORT', 5000)),
        debug=debug_mode
    )