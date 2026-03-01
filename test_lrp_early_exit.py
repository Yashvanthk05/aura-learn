import requests
import json

url = "http://127.0.0.1:8000/api/v1/explain/extractive"
payload = {
    "document_id": "24d08a4e-6130-47ec-9b6f-926cb8ebbb53",
    "chunk_ids": [7],
    "num_sentences": 3,
    "generate_lrp": True
}

headers = {
    "Content-Type": "application/json"
}

try:
    response = requests.post(url, json=payload, headers=headers)
    print("Status Code:", response.status_code)
    
    if response.status_code == 200:
        data = response.json()
        print(f"Explanation Methods: {data.get('explanation_methods')}")
        
        lrp = data.get("lrp_explanation")
        if lrp:
            print("\nLRP Explanation returned successfully!")
            print(f"Number of selected sentences: {len(lrp['selected_sentences'])}")
            print(f"Number of input sentences: {lrp['input_sentences']}")
            print(f"Dimensions of feature attribution matrix: {len(lrp['feature_attributions'])}x{len(lrp['feature_attributions'][0]) if lrp['feature_attributions'] else 0}")
            print(f"\nSample of first row (Top sentence attribution mapping):")
            print(lrp['feature_attributions'][0][:5], "...")
        else:
            print("\nLRP block is missing from response!")
            print(json.dumps(data, indent=2))
            
    else:
        print("Response Body:")
        print(response.text)
except requests.exceptions.ConnectionError:
    print("Could not connect to the server. Is it running?")
