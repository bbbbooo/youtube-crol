from operator import pos
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import re
import urllib.request
import os
import openpyxl as op
import plotly.express as px
import plotly.graph_objects as go
from konlpy.tag import Okt
from pyparsing import col
from sqlalchemy import column
from tqdm import tqdm
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.layers import Embedding, Dense, LSTM
from keras.models import Sequential
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
import streamlit as st
import pickle
matplotlib.use('TkAgg')   


st.header('asdsad')

tokenizer = Tokenizer()
okt = Okt()

PATH = '/Users/82102/Desktop/project/yt_cr/model/save_model/'
loaded_model = load_model(PATH + 'best_model.h5')

PATH2 = '/Users/82102/Desktop/project/yt_cr/model/token/'
with open(PATH2+'tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']
max_len = 30


#sentence = input('감성분석할 문장을 입력해 주세요.: ')

contain = []
contain_number = []
contain2 = []
contain2_number = []

def sentiment_predict(new_sentence):
        new_sentence = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣 ]','', str(new_sentence))
        new_sentence = okt.morphs(new_sentence, stem=True) # 토큰화
        new_sentence = [word for word in new_sentence if not word in stopwords] # 불용어 제거
        encoded = tokenizer.texts_to_sequences([new_sentence]) # 정수 인코딩
        pad_new = pad_sequences(encoded, maxlen = max_len) # 패딩
        score = float(loaded_model.predict(pad_new)) # 예측
        
        if(score > 0.5):
            contain.append(list)
            contain_number.append(score * 100)     
        else:
            contain2.append(list)
            contain2_number.append( (1 - score) * 100)
            
          

filename = pd.read_excel('/Users/82102/Desktop/project/yt_cr/video_xlxs/파리지옥.xlsx')
sheet = filename['comment']

# comment 칼럼의 각각의 데이터를 읽기
for cell in sheet:
    list = []
    output_sentence = str(cell)
    #cmrp = re.sub('[^가-힣]', '', str(cell))
    if "</a>" in output_sentence:
        split = output_sentence.split('</a>')
        if split[1] == '':
            continue
        else:
            list.append(split[1])
    else:
        list.append(output_sentence)
    
    sentiment_predict(list)
    
def Sub_comments():
    cmrp = re.sub('[^가-힣]', '', sheet)
    return cmrp 

#긍정
pd_contain = pd.DataFrame({'Postive' : contain})
pd_contain_number = pd.DataFrame({'postive-comments': contain_number})
result = pd.concat([pd_contain, pd_contain_number], axis=1)

#부정
pd_contain2 = pd.DataFrame({'Negative' : contain2})
pd_contain_number2 = pd.DataFrame({'negative-comments': contain2_number})
result2 = pd.concat([pd_contain2, pd_contain_number2], axis=1)

show_chart = pd.concat([result, result2], axis=1)

st.write(result)
st.write(result2)

def Create_plot():
  allen = len(sheet)
  poslen = len(pd_contain)
  neglen = len(pd_contain2)
  
  pos_ratio = (poslen/allen) * 100
  neg_ratio = (neglen/allen) * 100
  
  labels = ['Postive', 'Negative']
  ratio = pos_ratio, neg_ratio
  
  fig, ax = plt.subplots()
  ax.pie(ratio, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
  ax.axis('equal')
  
  st.pyplot(fig)

# pie plot
st.header('Pie Plot')
Create_plot()