from flask import Flask, jsonify
from flask_cors import CORS
from src.routes.spam import spam_bp 
from src.routes.sentiment import sentiment_bp

def create_app():
    app = Flask(__name__)

    # CORS config
    CORS(app, resources={
        r"/api/*": {
            "origins": "*", 
            "allow_headers": ["Content-Type", "Authorization"],
            "methods": ["GET", "POST", "OPTIONS"]
        }
    })


    app.register_blueprint(spam_bp, url_prefix='/api')
    app.register_blueprint(sentiment_bp, url_prefix='/api')


    @app.route('/health', methods=['GET'])
    def health_check():
        print("Health check ping received.")
        return jsonify({'status': 'ok'}), 200
    
    return app

app = create_app()

if __name__ == '__main__':
    from waitress import serve
    serve(app, host='0.0.0.0', port=8080)