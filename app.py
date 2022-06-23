import pickle
from flask import Flask, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename
import os,soundfile,librosa,glob
import numpy as np

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

modelSVC = pickle.load(open('svc.pkl','rb'))
modelRF = pickle.load(open('rf.pkl','rb'))
modelMLP = pickle.load(open('mlp.pkl','rb'))

def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate
        if chroma:
            stft=np.abs(librosa.stft(X))
        result=np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result=np.hstack((result, mfccs))
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result=np.hstack((result, chroma))
        if mel:
            mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result=np.hstack((result, mel))
    return result

data = {'file':'','prediction':''}
@app.route('/', methods=['GET','POST'])
def home():
    if request.method == 'POST':
        data['file'] = ''
        data['prediction'] = ''

        #Saving file uploaded to /uploads
        file = request.files['file']
        filename = secure_filename(file.filename)
        save_file = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_file)
        
        #Prediction
        x,y = [],[]
        feature=extract_feature(save_file, mfcc=True, chroma=True, mel=True)
        predSVC = modelSVC.predict(np.array([feature]))
        predRF = modelRF.predict(np.array([feature]))
        predMLP = modelMLP.predict(np.array([feature]))
        data['file'],data['prediction']=str(filename),[predSVC[0].upper(),predRF[0].upper(),predMLP[0].upper()]
        
        #After prediction the file gets deleted from /uploads
        os.remove(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        
    return render_template('website.html')

@app.route('/result')
def result():
    return render_template('result.html',filename=data['file'],pred=data['prediction'])

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000,debug=True)