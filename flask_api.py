import json
import math
import librosa
import os
import numpy as np



'''
#from keras.models import load_model
Note: Don't import keras directly as your model is saved with Tensorflow's keras high level api.
'''
from tensorflow.keras.models import load_model


# Flask utils
from flask import Flask, redirect, url_for, request, render_template,jsonify
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer



DATASET_PATH = "dataset/"
JSON_PATH = "test_data.json"
SAMPLE_RATE = 22050
TRACK_DURATION = 30 # measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION



def save_mfcc(file_path, json_path, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=10):

    """
    Extracts MFCCs from music dataset and saves them into a json file along with genre labels.
        :param dataset_path (str): Path to dataset
        :param json_path (str): Path to json file used to save MFCCs
        :param num_mfcc (int): Number of coefficients to extract
        :param n_fft (int): Interval we consider to apply FFT(Fast Fourier Transformation). Measured in # of samples
        :param hop_length (int): Sliding window for FFT. Measured in # of samples
        :param: num_segments (int): Number of segments we want to divide sample tracks into
        :return:
        """

    # dictionary to store mapping, labels, and MFCCs
    data = {
        "mfcc": [],
        "files": []
    }

    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

    signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)

    # process all segments of audio file
    for d in range(num_segments):
        # calculate start and finish sample for current segment
        start = samples_per_segment * d
        finish = start + samples_per_segment

        # extract mfcc
        mfcc = librosa.feature.mfcc( signal[start:finish], sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
        mfcc = mfcc.T

        # store only mfcc feature with expected number of vectors
        if len(mfcc) == num_mfcc_vectors_per_segment:
            data["mfcc"].append(mfcc.tolist())
            data['files'].append(file_path)
            break
        

    return data

def model_predict(file_path):
    #load the trained RNN model
    model=load_model('my_model.h5')      # Model saved with Keras model.save()

    print(model.summary())

    data=save_mfcc(file_path, JSON_PATH, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=10)
    
    print(np.array(data['mfcc']).shape)
    
    predicted=model.predict(np.array(data['mfcc']))
    
    
    maps={0:'blues',1:'classical',2:'country',3:'disco',4:'hiphop',5:'jazz',6:'metal',7:'pop',8:'reggae',9:'rock'}
    
    predicted_keyword=maps.get(np.argmax(predicted)) 
    
 
    
    return predicted_keyword


app = Flask(__name__)
@app.route("/")
def index():
    return render_template('test.html')

@app.route('/predict',methods=['GET', 'POST'])
def upload():
    if request.method=='POST':
        # get file from POST request and save it
        f = request.files['file']

        print("My output",f)

        print('1...',secure_filename(f.filename))

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        #Make prediction
        preds = model_predict(file_path)
        # print("prediction", preds)
    return render_template("test.html", suc_msg="File successfully uploaded!!!",test=preds)

if __name__=='__main__':
    app.run()
