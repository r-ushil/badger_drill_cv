from os import environ
from flask import Flask

from cover_drive_blueprint import cover_drive_blueprint

app = Flask(__name__)

app.register_blueprint(cover_drive_blueprint)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(environ.get("PORT", 8080)))
