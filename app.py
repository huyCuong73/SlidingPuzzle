from flask import Flask,jsonify
from routers import router
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

router.router(app)

if __name__ == "__main__":
    app.run(debug=True)