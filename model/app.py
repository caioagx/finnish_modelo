from flask import Flask, request, jsonify
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

app = Flask(__name__)

# Load the trained model and tokenizer
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=10)
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Load the model weights
model.load_state_dict(torch.load('model_weights.pth', map_location=torch.device('cpu')))
model.eval()

# Verificar se os pesos foram carregados corretamente
print("Pesos do modelo carregados com sucesso")

# Define the API endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # Check if the request has the correct Content-Type header
    if request.headers['Content-Type'] != 'application/json':
        return jsonify({'error': 'Invalid Content-Type'}), 400

    # Get the input data from the JSON request
    data = request.get_json()
    transaction_descriptions = data['transaction_descriptions']

    # Preprocess the input transaction descriptions
    inputs = tokenizer(transaction_descriptions, padding=True, truncation=True, return_tensors='pt')

    # Make predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_labels = torch.argmax(logits, dim=1)

    # Adicionando nossas labels
    labels = [
        'Restaurants', 'Shopping', 'Gas', 'Entertainment', 'Groceries', 'Salary',
        'Interest Income', 'Utilities', 'Pharmacy', 'Transfer-Out'
    ]

    predicted_labels = [labels[label] for label in predicted_labels]

    # Prepare the response JSON object
    response = []
    for desc, label in zip(transaction_descriptions, predicted_labels):
        response.append({'transaction_description': desc, 'predicted_label': label})

    # Return the JSON response
    return jsonify(response)

# Função de teste local
def local_test():
    test_descriptions = [
        "Bought groceries", 
        "Monthly salary", 
        "Paid electricity bill"
    ]
    inputs = tokenizer(test_descriptions, padding=True, truncation=True, return_tensors='pt')

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_labels = torch.argmax(logits, dim=1)

    labels = [
        'Restaurants', 'Shopping', 'Gas', 'Entertainment', 'Groceries', 'Salary',
        'Interest Income', 'Utilities', 'Pharmacy', 'Transfer-Out'
    ]

    predicted_labels = [labels[label] for label in predicted_labels]

    for desc, label in zip(test_descriptions, predicted_labels):
        print({'transaction_description': desc, 'predicted_label': label})

# Chamar a função de teste local
local_test()



# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
