import base64
import numpy as np
from PIL import Image
from io import BytesIO
from flask import Flask, request, jsonify, json, make_response
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

BASE_MODEL_ULCERNOSIS = '/home/ubuntu/ulcernosis-v4.h5'




app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    # Carga la imagen enviada desde el formulario
    print(BASE_MODEL_ULCERNOSIS)
    model = load_model(BASE_MODEL_ULCERNOSIS, compile=False)
    image = request.files['image']
    img = Image.open(image)
    # Preprocesa la imagen para que sea compatible con el modelo
    img = img.resize((224, 224))
    img = np.array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    # Realiza la predicción utilizando el modelo
    prediction = model.predict(img)
    percent_of_prediction_stage_1 = "{:.9f}".format(prediction[0][0]*100)
    percent_of_prediction_stage_2 = "{:.9f}".format(prediction[0][1]*100)
    percent_of_prediction_stage_3 = "{:.9f}".format(prediction[0][2]*100)
    percent_of_prediction_stage_4 = "{:.9f}".format(prediction[0][3]*100)
    stage_predicted = np.argmax(prediction)
    results = {
            'stage_1':percent_of_prediction_stage_1,
            'stage_2':percent_of_prediction_stage_2,
            'stage_3':percent_of_prediction_stage_3,
            'stage_4':percent_of_prediction_stage_4,
            'stage_predicted': str(stage_predicted+1)
        }
    
    # TODO: CREAR UNA CLASE PARA ENVIAR LAS PROBABILIDADES DE CADA ETAPA Y LA ETAPA PREDECIDA
    # Devuelve la predicción como una respuesta HTTP
    return make_response(jsonify(results))
    
@app.route('/')
def main():
    # Carga la imagen enviada desde el formulario
    return 'Hola Mundo'
    


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
