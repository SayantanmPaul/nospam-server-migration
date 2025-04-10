from flask import Flask, jsonify
from flask_cors import CORS
from app.routes.spam import spam_bp 

app = Flask(__name__)
CORS(app, resources={r"/predict-spam": {"origins": "*", "allow_headers": "*", "methods": ["POST"]}})

app.register_blueprint(spam_bp)

@app.route('/health', methods=['GET'])
def health_check():
    print("Health check ping received.")
    return jsonify({'status': 'ok'}), 200

if __name__ == '__main__':
    from waitress import serve
    serve(app, host='0.0.0.0', port=8080)