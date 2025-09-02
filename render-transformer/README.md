# Sentence Transformer API on Render

This project provides a REST API for generating sentence embeddings using the `all-MiniLM-L6-v2` model from Sentence Transformers, deployed on Render.com.

## Overview

The API converts text sentences into high-dimensional vector embeddings that can be used for:
- Semantic similarity search
- Text clustering
- Document retrieval
- Machine learning feature extraction

## Features

- **Fast inference**: Optimized for quick embedding generation
- **Flexible input formats**: Supports both map-based and KServe-compatible formats
- **CORS enabled**: Ready for web applications
- **Health monitoring**: Built-in health check endpoint
- **Production ready**: Uses Gunicorn WSGI server

## API Endpoints

### Health Check
```
GET /health
```
Returns service status and model information.

**Response:**
```json
{
  "status": "healthy",
  "model": "all-MiniLM-L6-v2"
}
```

### Generate Embeddings
```
POST /embedding
```

#### Input Format 1: Map-based (ID to text mapping)
```json
{
  "id1": "This is the first sentence.",
  "id2": "Here is another sentence."
}
```

**Response:**
```json
{
  "id1": "[-0.010852468200027943, -0.016728922724723816, ...]",
  "id2": "[0.023456789012345678, -0.034567890123456789, ...]"
}
```

#### Input Format 2: KServe-compatible
```json
{
  "instances": [
    "This is the first sentence.",
    "Here is another sentence."
  ]
}
```

**Response:**
```json
{
  "predictions": [
    [-0.010852468200027943, -0.016728922724723816, ...],
    [0.023456789012345678, -0.034567890123456789, ...]
  ]
}
```

## Local Development

### Prerequisites
- Python 3.8+
- pip

### Setup
1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Pre-download the model:
   ```bash
   python getmodels.py
   ```

4. Run the development server:
   ```bash
   python app.py
   ```

The API will be available at `http://localhost:5000`

### Testing Locally
```bash
# Health check
curl http://localhost:5000/health

# Generate embeddings (map format)
curl -X POST http://localhost:5000/embedding \
  -H "Content-Type: application/json" \
  -d '{"text1": "Hello world", "text2": "How are you?"}'

# Generate embeddings (KServe format)
curl -X POST http://localhost:5000/embedding \
  -H "Content-Type: application/json" \
  -d '{"instances": ["Hello world", "How are you?"]}'
```

## Deployment on Render

### Quick Deploy
1. Fork this repository
2. Connect your GitHub account to Render
3. Create a new Web Service
4. Connect your forked repository
5. Configure the service:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn --bind 0.0.0.0:$PORT --workers 1 --timeout 120 app:app`
   - **Environment**: `Docker` (if using Dockerfile) or `Python 3`

### Using Docker (Recommended)
The included Dockerfile is optimized for Render deployment:

```dockerfile
FROM python:3.8-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 10000
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "--workers", "1", "--timeout", "120", "app:app"]
```

### Environment Variables
No special environment variables are required. The app will:
- Use `PORT` environment variable (automatically set by Render)
- Download the model on first startup if not pre-installed
- Cache models in `/tmp/transformers_cache`

### Performance Considerations
- **Memory**: The service requires ~2GB RAM for the model
- **CPU**: Single worker recommended to avoid memory issues
- **Startup time**: First request may take 30-60 seconds for model download
- **Timeout**: Set to 120 seconds to handle model loading

## Model Information

- **Model**: `all-MiniLM-L6-v2`
- **Dimensions**: 384
- **Max Sequence Length**: 256 tokens
- **Use Cases**: General purpose sentence embeddings
- **Performance**: Good balance of speed and quality

## Error Handling

The API returns appropriate HTTP status codes:
- `200`: Success
- `400`: Bad request (missing or invalid data)
- `422`: Unprocessable entity (invalid input format)
- `500`: Internal server error

## CORS Support

The API includes CORS headers for cross-origin requests:
- `Access-Control-Allow-Origin: *`
- `Access-Control-Allow-Methods: OPTIONS,POST,GET`
- `Access-Control-Allow-Headers: Content-Type,X-Api-Key`

## Migration from AWS Lambda

This API maintains compatibility with the original AWS Lambda function:
- Same input/output formats
- Same model (`all-MiniLM-L6-v2`)
- Same embedding dimensions
- CORS headers preserved

### Key Differences
- **Protocol**: HTTP REST API instead of Lambda event
- **Deployment**: Render instead of AWS
- **Scaling**: Single instance instead of auto-scaling
- **Authentication**: No API key required (can be added if needed)

## Troubleshooting

### Model Download Issues
If the model fails to download on startup:
1. Check internet connectivity
2. Verify Hugging Face Hub access
3. Pre-download using `getmodels.py`
4. Check disk space (model ~90MB)

### Memory Issues
If you encounter out-of-memory errors:
1. Reduce batch size in requests
2. Use single worker configuration
3. Monitor Render service logs
4. Consider upgrading to higher memory tier

### Slow Response Times
- First request after deployment will be slow (model loading)
- Subsequent requests should be fast (<1 second)
- Consider pre-warming with a health check

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test locally
5. Submit a pull request

## License

This project is open source and available under the [MIT License](LICENSE).
