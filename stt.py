import speech_recognition as sr
import os
import sys
import streamlit as st
import pickle
import re
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from konlpy.tag import Okt
from pydub import AudioSegment
from keras.models import load_model


#-----------------------------------------------------------------------------------



# 업로드
def upload():
    uploaded_file = st.file_uploader("Choose a file", type=(["mp3", "wav"]))
    if uploaded_file is not None:
        #bytes_data = 오디오 파일
        global bytes_data
        bytes_data = uploaded_file.name
        st.success('파일을 업로드 했습니다. : {} '.format(bytes_data))


# 음성 파일 불러와서 텍스트로 전환
def STT():
    r = sr.Recognizer()
    # 파일명과 확장자 분리
    global name
    name, ext = os.path.splitext(filename)

    # wav
    if ext == ".wav":
        harvard_audio = sr.AudioFile(filepath)
        with harvard_audio as source:
            audio = r.record(source)
        global text
        text = r.recognize_google(audio, language='ko-KR')
    # mp3
    elif ext == '.mp3':
        mp3_sound = AudioSegment.from_mp3(filepath)
        wav_sound = mp3_sound.export("{0}.wav".format(name), format="wav")
        harvard_audio = sr.AudioFile(wav_sound)
        with harvard_audio as source:
            audio = r.record(source, duration=150)
        text = r.recognize_google(audio, language='ko-KR')
    # 나머지..
    else:
        st.write("wav 와 mp3 형식만 호환됩니다.")


contain = []  # 긍정 cell
contain_number = []
contain2 = []  # 부정 cell
contain2_number = []


def Analysis():
    tokenizer = Tokenizer()
    okt = Okt()
    max_len = 30

    # 불용어
    stopwords = ['은', '는', '이', '가', '하', '아', '것', '들', '의', '있', '되', '수', '보', '주', '등', '한', '줄', '를', '을', '에', '에게', '께', '한테', '더러', '에서', '에게서',
                 '한테서', '로', '으로', '와', '과', '도', '부터', '도', '만', '이나', '나', '라도', '의', '거의', '겨우', '결국', '그런데', '즉', '참', '챗', '할때', '할뿐', '함께', '해야한다', '휴']

    #PATH = '/Users/82102/Desktop/project/yt_cr/backup/'
    PATH = '/Users/82102/Desktop/project/yt_cr/model_test/'
    #PATH2 = '/Users/82102/Desktop/project/yt_cr/backup/'
    PATH2 = '/Users/82102/Desktop/project/yt_cr/model_test/'

    #모델 및 토큰 불러오기
    model = load_model(PATH + 'best_model.h5')
    with open(PATH2+'tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    # 감정 예측

    def sentiment_predict(new_sentence):
        new_sentence = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣 ]', '', str(new_sentence))
        new_sentence = okt.morphs(new_sentence, stem=True)  # 토큰화
        new_sentence = [word for word in new_sentence if not word in stopwords]  # 불용어 제거
        encoded = tokenizer.texts_to_sequences([new_sentence])  # 정수 인코딩
        pad_new = pad_sequences(encoded, maxlen=max_len)  # 패딩
        score = float(model.predict(pad_new))  # 예측

        # 긍정적이라면 contain 리스트에 추가
        if(score > 0.5):
            contain.append(text)
            contain_number.append(score * 100)

        # 부정적이라면 contain2 리스트에 추가
        else:
            contain2.append(text)
            contain2_number.append((1 - score) * 100)

    sentiment_predict(text)


#---------------------------------------------------------------------------
st.header('Text To Speech')
st.container()
upload()
path = '/Users/82102/Desktop/project/yt_cr/audio/'
filename = bytes_data
filepath = path + bytes_data
STT()
Analysis()

st.header('긍정')
st.write(contain, contain_number)

st.header('부정')
st.write(contain2, contain2_number)
