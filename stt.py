from matplotlib.text import Text
import speech_recognition as sr
import os
import sys
import streamlit as st
import pickle
import re
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from konlpy.tag import Okt
from pydub import AudioSegment
from keras.models import load_model
from operator import truediv
from pykospacing import Spacing

contain = []  # 긍정 cell
contain_number = []
contain2 = []  # 부정 cell
contain2_number = []

def Analysis(sentence):
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
            contain.append(sentence)
            contain_number.append(score*100)

        # 부정적이라면 contain2 리스트에 추가
        else:
            contain2.append(sentence)
            contain2_number.append((1 -score)*100)

    sentiment_predict(sentence)

def record():
    if st.button('녹음'):
        # if st.button('종료', key=1):
        #     raise IndexError
        
        con = st.container()
        r=sr.Recognizer()
        with sr.Microphone() as source:
            print("Say something!")
            st.write("음성 녹음이 활성화 됐습니다.")
            audio=r.listen(source)
            

        try:
            transcript=r.recognize_google(audio, language="ko-KR")
            print("Google Speech Recognition thinks you said "+transcript)
            
            con.caption("Sentence")
            con.write(transcript)
            
            Analysis(transcript)
            
            pd_contain = pd.DataFrame({'긍정 문장' : contain})
            pd_contain_number = pd.DataFrame({'확률': contain_number})
            pos_result = pd.concat([pd_contain, pd_contain_number], axis=1)
        
            pd_contain2 = pd.DataFrame({'부정 문장' : contain2})
            pd_contain_number2 = pd.DataFrame({'확률': contain2_number})
            neg_result = pd.concat([pd_contain2, pd_contain_number2], axis=1)
            
            st.header('긍정')
            st.table(pos_result)
            
            st.header('부정')
            st.table(neg_result)
            
            
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
            st.write("음성을 인식하지 못했습니다.")
        except sr.RequestError as e:
            print("Could not request results from Google Speech Recognition service; {0}".format(e))
            st.write("예상치 못한 오류가 발생했습니다. {0}".format(e))
            
        if st.sidebar.button('파일 저장', key=2):
            with open("./audio/"+"Record.wav", "wb") as f:
                f.write(audio.get_wav_data())
            
record()

