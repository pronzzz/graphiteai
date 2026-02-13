from flask import Flask, request, send_file
from flask_cors import CORS
from PIL import Image
import io
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.inference import sketch_generator

app = Flask(__name__)
CORS(app)

@app.route('/sketch', methods=['POST'])
def generate_sketch():
    if 'image' not in request.files:
        return "No image provided", 400
    
    file = request.files['image']
    try:
        image = Image.open(file.stream).convert("RGB")
        sketch = sketch_generator.predict(image)
        
        # Determine output format (JPEG or PNG based on input or default to PNG)
        img_io = io.BytesIO()
        sketch.save(img_io, 'PNG')
        img_io.seek(0)
        
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        return str(e), 500

@app.route('/', methods=['GET'])
def health_check():
    status = "Active" if sketch_generator.model_loaded else "Fallback (OpenCV)"
    return f"GraphiteAI Backend Running. Mode: {status}"

if __name__ == '__main__':
    app.run(debug=True, port=5001)
