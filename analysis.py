import os, glob, pickle
import librosa
import soundfile
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

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

emotions={
  '01':'anger',
  '02':'sad',
  '03':'surprise',
  '04':'happy',
  '05':'fear',
  '06':'neutral'
}

observed_emotions=['anger','sad','happy']

def load_data(test_size=0.2):
    x,y=[],[]
    for file in glob.glob("KData_Cleaned/*.wav"):
        file_name=os.path.basename(file)
        emotion=emotions[file_name.split("-")[1]]
        if emotion not in observed_emotions:
            continue
        feature=extract_feature(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(emotion)
    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)

x_train,x_test,y_train,y_test=load_data(test_size=0.3)
print(y_test)
y_predMLP=modelMLP.predict(x_test)
print(y_predMLP)
y_predRF=modelRF.predict(x_test)
print(y_predRF)
y_predSVC=modelSVC.predict(x_test)


accuracy=accuracy_score(y_true=y_test, y_pred=y_predMLP)
print("Accuracy(MLP): {:.2f}%".format(accuracy*100))
accuracy=accuracy_score(y_true=y_test, y_pred=y_predRF)
print("Accuracy(RF): {:.2f}%".format(accuracy*100))
accuracy=accuracy_score(y_true=y_test, y_pred=y_predSVC)
print("Accuracy(SVC): {:.2f}%".format(accuracy*100))