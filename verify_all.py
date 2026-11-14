import requests
import json
import time

BASE_URL = "http://127.0.0.1:8000"

def test_prediction():
    print("--- 1 & 2 & 3 & 4 & 5 & 6 & 12. Testing /predict with sample_mri_glioma.png ---")
    files = {'file': open('sample_images/sample_mri_glioma.png', 'rb')}
    data = {'model_type': 'mobilenet_v2'}
    
    t0 = time.time()
    response = requests.post(f"{BASE_URL}/predict", files=files, data=data)
    t1 = time.time()
    
    print(f"Status Code: {response.status_code}")
    if response.status_code != 200:
        print("Failed:", response.text)
        return None
        
    res = response.json()
    print("Quality Grade:", res['quality']['grade'])
    print("Quality Details (SNR/Sharpness):", res['quality']['snr_db'], res['quality']['sharpness_laplacian_var'])
    print("Prediction:", res['prediction'])
    print("Confidence %:", res['confidence_percentage'])
    print("Confidence Rating:", res['confidence_rating'])
    print("Probabilities:", dict(res['sorted_probabilities']))
    print("Grad-CAM overlays present:", "overlays" in res and "blended_url" in res["overlays"])
    print("ROI measurements present:", "measurements" in res and res["measurements"]["has_tumor_roi"])
    if res["measurements"]["has_tumor_roi"]:
        print("ROI Bounding Box:", res["measurements"]["bounding_box"])
        print("ROI Width/Height:", res["measurements"]["width_px"], "x", res["measurements"]["height_px"])
        print("Physical Calibrated:", res["measurements"]["physical_calibrated"])
        print("Calibration Note:", res["measurements"]["calibration_note"])
        
    print("Analysis ID:", res['analysis_id'])
    
    # Store history for comparison
    return res['analysis_id']

def test_manual_calibration():
    print("\n--- 6. Testing /predict with Manual Calibration ---")
    files = {'file': open('sample_images/sample_mri_glioma.png', 'rb')}
    data = {'model_type': 'mobilenet_v2', 'manual_pixel_spacing': '0.5'}
    response = requests.post(f"{BASE_URL}/predict", files=files, data=data)
    if response.status_code == 200:
        res = response.json()
        print("Physical Calibrated:", res["measurements"]["physical_calibrated"])
        print("Calibration Note:", res["measurements"]["calibration_note"])
    else:
        print("Failed to run manual calibration test.")

def test_history():
    print("\n--- 7. Testing /api/history ---")
    response = requests.get(f"{BASE_URL}/api/history")
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        history = response.json().get('history', [])
        print(f"History length: {len(history)}")
        if len(history) > 0:
            print("Latest entry keys:", list(history[0].keys()))
            print("Latest entry ID format:", history[0]['analysis_id'])

def test_compare(id1, id2):
    print("\n--- 8. Testing /api/compare ---")
    data = {'analysis_id_1': id1, 'analysis_id_2': id2}
    response = requests.post(f"{BASE_URL}/api/compare", data=data)
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        res = response.json()
        print("Comparison disclaimer:", res['comparison']['clinical_disclaimer'])
        print("Area Delta (px):", res['comparison']['delta_area_px'])

def test_report(analysis_id):
    print(f"\n--- 9. Testing /report/{analysis_id} ---")
    response = requests.get(f"{BASE_URL}/report/{analysis_id}")
    print(f"Status Code: {response.status_code}")
    print("Content Type:", response.headers.get('Content-Type'))
    if "NeuroSyntra AI" in response.text:
        print("Branding 'NeuroSyntra AI' found in report HTML.")

if __name__ == "__main__":
    print("Running Verification Tests...\n")
    id1 = test_prediction()
    if id1:
        test_manual_calibration()
        test_history()
        
        print("\nUploading another image to test comparison...")
        files = {'file': open('sample_images/sample_mri_meningioma.png', 'rb')}
        data = {'model_type': 'mobilenet_v2'}
        resp2 = requests.post(f"{BASE_URL}/predict", files=files, data=data)
        id2 = resp2.json()['analysis_id']
        
        test_compare(id1, id2)
        test_report(id1)
