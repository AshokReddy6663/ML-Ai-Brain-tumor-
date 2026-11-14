import requests

url = "http://127.0.0.1:8000/predict"
img_path = "sample_images/sample_mri_glioma.png"

with open(img_path, "rb") as f:
    files = {"file": (img_path, f, "image/png")}
    data = {"model_type": "mobilenet_v2"}
    response = requests.post(url, files=files, data=data)

print(f"Status Code: {response.status_code}")
if response.status_code == 200:
    res = response.json()
    print("Prediction:", res.get("prediction_display"))
    print("Confidence:", res.get("confidence_percentage"), "%")
    print("Probabilities:", res.get("probabilities"))
else:
    print("Response:", response.text)
