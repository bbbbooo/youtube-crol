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
matplotlib.use('TkAgg')   


st.header('asdsad')

tokenizer = Tokenizer()
okt = Okt()

PATH = '/Users/82102/Desktop/project/yt_cr/study_analy/model/'
loaded_model = load_model(PATH + 'best_model.h5')
#print("\n 테스트 정확도: %.4f" % (loaded_model.evaluate(X_test, y_test)[1]))

stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']
max_len = 30


#sentence = input('감성분석할 문장을 입력해 주세요.: ')

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
            # st.header("Positive")
            # st.text(cell)
            
            
            #print(cell)
            contain.append(cell)
            contain_number.append(score * 100)
            
            #print("{:.2f}% 확률로 긍정 리뷰입니다.\n".format(score * 100))
            
        else:
            # st.header("Negative")
            # st.text(cell)
            
            contain2.append(cell)
            contain2_number.append( (1 - score) * 100)
            
            #print("{:.2f}% 확률로 부정 리뷰입니다.\n".format((1 - score) * 100))
            
          
        
          
          

filename = pd.read_excel('/Users/82102/Desktop/project/yt_cr/video_xlxs/벨베스 사기네.xlsx')
sheet = filename['comment']



#sheet2 = sheet.replace('<a href="https://www.youtube.com/watch?v=%s&t=%dm%ds">%d:%d</a>' % (sentence, num_arr, num_arr, num_arr, num_arr), "" )
#sheet2 = sheet.replace("100만 축하드려용", "10만 축하드려용")
#sheet2[:5]
#print(sheet2)


# comment 칼럼의 각각의 데이터를 읽기
for cell in sheet:
    output_sentence = str(cell)
    sentiment_predict(output_sentence)
    
    
  

  

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
  ax.pie(ratio, labels=labels, autopct='%1.1f%%', shadow=True)
  ax.axis('equal')
  
  st.pyplot(fig)
  #fig = plt.pie(ratio, labels=labels, autopct='%1.1f%%')
  #plt.savefig("mygraph.png")

# pie plot
st.header('Pie Plot')
Create_plot()