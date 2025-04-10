from flask import Flask, jsonify
from flask_cors import CORS
from src.routes.spam import spam_bp 


def create_app():
    app = Flask(__name__)
    app.register_blueprint(spam_bp, url_prefix='/api')

    CORS(app, resources={r"/predict-spam": {"origins": "*", "allow_headers": "*", "methods": ["POST"]}})

    @app.route('/health', methods=['GET'])
    def health_check():
        print("Health check ping received.")
        return jsonify({'status': 'ok'}), 200
    
    return app

app = create_app()

if __name__ == '__main__':
    from waitress import serve
    serve(app, host='0.0.0.0', port=8080)