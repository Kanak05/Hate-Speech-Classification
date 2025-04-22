from flask import Flask, request, render_template
import pickle

# Load trained model and vectorizer
model = pickle.load(open("model.pkl", 'rb'))
vectorizer = pickle.load(open("vectorizer.pkl", 'rb'))

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/detect", methods=['POST'])
def detect():
    featur = request.form['feature']
    featur_vector = vectorizer.transform([featur]).toarray()
    prediction = model.predict(featur_vector)

    if prediction[0] == "Normal":
        output = "Normal"
    elif prediction[0] == "Hate speech":
        output = "Hate speech"
    else:
        output = "Offensive language"

    return render_template("index.html", message=output, input_text=featur)


if __name__ == "__main__":
    app.run(debug=True)
