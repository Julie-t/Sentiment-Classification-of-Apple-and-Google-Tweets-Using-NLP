import pickle
from flask import Flask, request, jsonify, render_template

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    sentiment = None
    if request.method == "POST":
        text = request.form["text"]
        sentiment = model.predict([text])[0]
    return render_template("index.html", sentiment=sentiment)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text")
    prediction = model.predict([text])[0]
    return jsonify({"text": text, "sentiment": prediction})

if __name__ == "__main__":
    app.run(debug=True)