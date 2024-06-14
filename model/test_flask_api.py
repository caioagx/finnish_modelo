import requests
import json

# Definir a URL da API
url = "http://127.0.0.1:5000/predict"

# Definir os dados de entrada para o teste
data = {
    "transaction_descriptions": [
        "Bought groceries",
        "Monthly salary",
        "Paid electricity bill",
        "Gas station purchase",
        "Movie tickets",
        "Online shopping",
        "Pharmacy purchase",
        "Transferred to savings",
        "Received interest",
        "Restaurant dinner",
        "Purchased electronics",
        "Filled gas tank",
        "Dinner at fast food",
        "Car repair",
        "Gym membership",
        "Book purchase",
        "Gift purchase",
        "Insurance payment"
    ]
}

# Definir os headers para a requisição
headers = {"Content-Type": "application/json"}

# Enviar a requisição POST para a API
response = requests.post(url, data=json.dumps(data), headers=headers)

# Imprimir a resposta da API
print(response.json())
