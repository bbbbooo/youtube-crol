
from imp import load_module
from keras.models import load_model
import keras
import tensorflow as tf
import numpy as np
import pandas as pd
import re
import json
from konlpy.tag import Okt
from keras.utils import pad_sequences
from keras.preprocessing.text import Tokenizer
import pickle


okt = Okt()
tokenizer  = Tokenizer()

DATA_CONFIGS = 'data_configs.json'
prepro_configs = json.load(open('/Users/82102/Desktop/project/yt_cr/study_test_analy/Users/82102/Desktop/project/yt_cr/test_analy/CLEAN_DATA/'+DATA_CONFIGS,'r'))


#PATH = 'C:/Users/82102/Desktop/project/yt_cr/study_test_analy/token/'
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

word_vocab = tokenizer.word_index #단어사전형태
prepro_configs['vocab'] = word_vocab

tokenizer.fit_on_texts(word_vocab)

MAX_LENGTH = 8 #문장최대길이

def sentiment_predict(sentence):
  sentence = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣\\s ]','', sentence)
  stopwords = ['은','는','이','가','하','아','것','들','의','있','되','수','보','주','등','한'] # 불용어 추가할 것이 있으면 이곳에 추가
  sentence = okt.morphs(sentence, stem=True) # 토큰화
  sentence = [word for word in sentence if not word in stopwords] # 불용어 제거
  vector  = tokenizer.texts_to_sequences(sentence)
  pad_new = pad_sequences(vector, maxlen = MAX_LENGTH) # 패딩


  model = keras.models.load_model('/Users/82102/Desktop/project/yt_cr/study_test_analy/DATA_OUT/cnn_classifier_kr/')
  model.load_weights('/Users/82102/Desktop/project/yt_cr/study_test_analy/DATA_OUT/cnn_classifier_kr/weights.h5') #모델 불러오기

  predictions = model.predict(pad_new)
  predictions = float(predictions.squeeze(-1)[1])

  if(predictions > 0.5):
    print(cell)
    print("{:.2f}% 확률로 긍정 리뷰입니다.\n".format(predictions * 100))
  else:
    print(cell)
    print("{:.2f}% 확률로 부정 리뷰입니다.\n".format((1 - predictions) * 100))
    
    
# 다른 함수에서도 쓰기 위해 global 선언
global filename, sheet
filename = pd.read_excel('/Users/82102/Desktop/project/yt_cr/video_xlxs/파도파도 전자기기만 나오네;; 잇섭 못지않은 테크덕후 PD님들의 가방을 털어봤습니다.xlsx')
sheet = filename['comment']
# comment 칼럼의 각각의 데이터를 읽기

for cell in sheet:
    output_sentence = str(cell)
    sentiment_predict(output_sentence)