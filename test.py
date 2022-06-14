import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import urllib.request
import os
import openpyxl
from konlpy.tag import Okt
from tqdm import tqdm
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.layers import Embedding, Dense, LSTM
from keras.models import Sequential
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint




tokenizer = Tokenizer()
okt = Okt()

PATH = '/Users/82102/Desktop/project/yt_cr/study_analy/model/'
loaded_model = load_model(PATH + 'best_model.h5')
#print("\n 테스트 정확도: %.4f" % (loaded_model.evaluate(X_test, y_test)[1]))

stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']
max_len = 30

def sentiment_predict(new_sentence):
  new_sentence = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣 ]','', new_sentence)
  new_sentence = okt.morphs(new_sentence, stem=True) # 토큰화
  new_sentence = [word for word in new_sentence if not word in stopwords] # 불용어 제거
  encoded = tokenizer.texts_to_sequences([new_sentence]) # 정수 인코딩
  pad_new = pad_sequences(encoded, maxlen = max_len) # 패딩
  score = float(loaded_model.predict(pad_new)) # 예측
  
  
  if(score > 0.5):
    #print(new_sentence)
    print("{:.2f}% 확률로 긍정 리뷰입니다.\n".format(score * 100))
    
  else:
    #print(new_sentence)
    print("{:.2f}% 확률로 부정 리뷰입니다.\n".format((1 - score) * 100))

def sentiment_predict(new_sentence):
    for cell in sheet:
        new_sentence = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣 ]','', str(new_sentence))
        new_sentence = okt.morphs(new_sentence, stem=True) # 토큰화
        new_sentence = [word for word in new_sentence if not word in stopwords] # 불용어 제거
        encoded = tokenizer.texts_to_sequences([new_sentence]) # 정수 인코딩
        pad_new = pad_sequences(encoded, maxlen = max_len) # 패딩
        score = float(loaded_model.predict(pad_new)) # 예측
        
        if(score > 0.5):
            print(cell)
            print("{:.2f}% 확률로 긍정 리뷰입니다.\n".format(score * 100))
            
        else:
            print(cell)
            print("{:.2f}% 확률로 부정 리뷰입니다.\n".format((1 - score) * 100))
          
          

filename = pd.read_excel('/Users/82102/Desktop/project/yt_cr/video_xlxs/가렌으로 절대 이길 수 없는 상성.xlsx')
sheet = filename['comment']

# comment 칼럼의 각각의 데이터를 읽기
for cell in sheet:
    output_sentence = str(cell)
    sentiment_predict(output_sentence)