from flask import Flask

app = Flask(__name__)

@app.route('/')
def welcome():
    return "Hi akshada this side hiiii"

@app.route('/members')
def members():
    return "We are in members function attention"

if __name__ == '__main__':
    app.run(debug=True)
    