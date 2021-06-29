import os
import numpy as np
import json
import joblib

def init():
    print("This is init")
    global model
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'),'model.pkl')
    model = joblib.load(model_path)

def run(data):
    test = json.loads(data)
    print(f"received data {test}")
    try:
        data = np.array(json.loads(data))
        result = model.predict(data)
        return result.tolist()
    except Exception as err:
        return str(err)