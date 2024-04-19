from flask import Flask, render_template, request, jsonify
import torch
from transformers import XLNetForSequenceClassification, XLNetTokenizer

app = Flask(__name__)

# Load the XLNet model and tokenizer
model_path = "xlnet_model.pth"
model = torch.load(model_path, map_location=torch.device('cpu'))
tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')

@app.route("/")
def index():
    # Render the template named "index.html" located in the "templates" directory
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data['text']
    
    # Tokenize input text
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    
    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get predicted class
    predicted_class = torch.argmax(outputs.logits).item()
    
    return jsonify({'predicted_class': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)
