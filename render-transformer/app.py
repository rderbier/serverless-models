import json
import os
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer

# Set cache directory for transformers
os.environ['TRANSFORMERS_CACHE'] = '/tmp'

app = Flask(__name__)

# Load the model - will be downloaded on first run or loaded from ./model if available
try:
    # Try to load from local model directory first (if pre-downloaded)
    model = SentenceTransformer('./model')
    print("Loaded model from local ./model directory")
except:
    # Fallback to downloading the model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Downloaded and loaded model from Hugging Face")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for Render"""
    return jsonify({"status": "healthy", "model": "all-MiniLM-L6-v2"})

@app.route('/embedding', methods=['POST'])
def generate_embeddings():
    """
    Generate embeddings for input sentences.
    Supports two input formats:
    1. Map format: {"id1": "text1", "id2": "text2"}
    2. KServe format: {"instances": ["text1", "text2"]}
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        reply_kserve = False
        
        if isinstance(data, dict):
            if 'instances' in data:
                print("Input contains 'instances' - using KServe format")
                reply_kserve = True
                sentences = data['instances']
            else:
                print("Input is a map of IDs and sentences")
                reply_kserve = False
                sentences = [data[key] for key in data]
        else:
            return jsonify({"error": "Invalid input format"}), 422
        
        if not sentences:
            return jsonify({"error": "No sentences provided"}), 400
            
        # Generate embeddings
        embeddings = model.encode(sentences)
        
        # Format response based on input format
        if reply_kserve:
            response = {"predictions": embeddings.tolist()}
        else:
            keylist = list(data.keys())
            response = {keylist[i]: json.dumps(embeddings[i].tolist()) for i in range(len(embeddings))}
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return jsonify({"error": f"Processing error: {str(e)}"}), 500

@app.route('/embedding', methods=['OPTIONS'])
def handle_options():
    """Handle CORS preflight requests"""
    response = jsonify({})
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,X-Api-Key')
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Methods', 'OPTIONS,POST,GET')
    return response

@app.after_request
def after_request(response):
    """Add CORS headers to all responses"""
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,X-Api-Key')
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Methods', 'OPTIONS,POST,GET')
    return response

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
