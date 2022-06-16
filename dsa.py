from csv import excel
from tkinter import CENTER, Label
import streamlit as st
import pandas as pd
from googleapiclient.discovery import build
import warnings # 경고창 무시
import re
import os
import time
import pafy as pa
warnings.filterwarnings('ignore')

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


#---------------

st.title("YouTube 댓글 민심 사이트")
input_name = st.text_input(label="여기에 URL을 입력하세요", value="")

url= ""
url=input_name
my_str = url.replace("https://www.youtube.com/watch?v=","")


def youtubetitle():
    #제목 가져오기
    videoinfo = pa.new(url)
    video_title = videoinfo.title

    #제목 특수기호 있으면 공백으로 치환
    rp_video_title = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…《\》]', '', video_title)
    return rp_video_title

    
def Crawling():

    comments = list()
    api_obj = build('youtube', 'v3', developerKey='AIzaSyA3vPhjSHzj_mz4SSsu55eHZw0oydLA8fg')
    response = api_obj.commentThreads().list(part='snippet,replies', videoId=my_str, maxResults=100).execute()

    while response:
        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']
            comments.append([comment['textDisplay'], comment['authorDisplayName'], comment['publishedAt'], comment['likeCount']])

        if 'nextPageToken' in response:
            response = api_obj.commentThreads().list(part='snippet,replies', videoId=my_str, pageToken=response['nextPageToken'], maxResults=100).execute()
        else:
            break

    df = pd.DataFrame(comments)
    df.to_excel('%s.xlsx'%youtubetitle(), header=['comment', 'author', 'date', 'num_likes'], index=None)

    path = '%s.xlsx' % (youtubetitle())
    while os.path.exists(path) :
      df.to_excel('%s.xlsx' % (youtubetitle()), header=['comment', 'author', 'date', 'num_likes'], index=None)
      break

contain = []        #긍정 cell
contain_number =[]  #긍정 확률
contain2 = []       #부정 cell
contain2_number = []#부정 확률

def Analysis():
        # 데이터셋 다운로드. 완료하였다면 주석처리 할 것.
    # urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt", filename="ratings_train.txt")
    # urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt", filename="ratings_test.txt")

    train_data = pd.read_table('C:/Users/jaehoon/Desktop/00_study/Excel/DATA/ratings_train.txt')
    test_data = pd.read_table('C:/Users/jaehoon/Desktop/00_study/Excel/DATA/ratings_test.txt')

    # document 열과 label 열의 중복을 제외한 값의 개수
    train_data['document'].nunique(), train_data['label'].nunique()

    # document 열의 중복 제거
    train_data.drop_duplicates(subset=['document'], inplace=True)
    train_data['label'].value_counts().plot(kind = 'bar')

    #널값을 가진 샘플이 어디 인덱스에 위치했는지..
    train_data = train_data.dropna(how = 'any') # Null 값이 존재하는 행 제거

    # 한글과 공백을 제외하고 모두 제거
    train_data['document'] = train_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
    train_data['document'] = train_data['document'].str.replace('^ +', "") # white space 데이터를 empty value로 변경
    train_data['document'].replace('', np.nan, inplace=True)
    train_data = train_data.dropna(how = 'any')

    # test도 똑같이데이터 전처리
    test_data.drop_duplicates(subset = ['document'], inplace=True) # document 열에서 중복인 내용이 있다면 중복 제거
    test_data['document'] = test_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","") # 정규 표현식 수행
    test_data['document'] = test_data['document'].str.replace('^ +', "") # 공백은 empty 값으로 변경
    test_data['document'].replace('', np.nan, inplace=True) # 공백은 Null 값으로 변경
    test_data = test_data.dropna(how='any') # Null 값 제거

    #불용어 처리
    stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']

    okt = Okt()

    X_train = []
    for sentence in tqdm(train_data['document']):
        tokenized_sentence = okt.morphs(sentence, stem=True) # 토큰화
        stopwords_removed_sentence = [word for word in tokenized_sentence if not word in stopwords] # 불용어 제거
        X_train.append(stopwords_removed_sentence)


    X_test = []
    for sentence in tqdm(test_data['document']):
        tokenized_sentence = okt.morphs(sentence, stem=True) # 토큰화
        stopwords_removed_sentence = [word for word in tokenized_sentence if not word in stopwords] # 불용어 제거
        X_test.append(stopwords_removed_sentence)


    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train)

    threshold = 3
    total_cnt = len(tokenizer.word_index) # 단어의 수
    rare_cnt = 0 # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
    total_freq = 0 # 훈련 데이터의 전체 단어 빈도수 총 합
    rare_freq = 0 # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

    # 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
    for key, value in tokenizer.word_counts.items():
        total_freq = total_freq + value

        # 단어의 등장 빈도수가 threshold보다 작으면
        if(value < threshold):
            rare_cnt = rare_cnt + 1
            rare_freq = rare_freq + value

    # 전체 단어 개수 중 빈도수 2이하인 단어는 제거.
    # 0번 패딩 토큰을 고려하여 + 1
    vocab_size = total_cnt - rare_cnt + 1

    tokenizer = Tokenizer(vocab_size) 
    tokenizer.fit_on_texts(X_train)
    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)

    y_train = np.array(train_data['label'])
    y_test = np.array(test_data['label'])

    drop_train = [index for index, sentence in enumerate(X_train) if len(sentence) < 1]

    # 빈 샘플들을 제거
    X_train = np.delete(X_train, drop_train, axis=0)
    y_train = np.delete(y_train, drop_train, axis=0)


    def below_threshold_len(max_len, nested_list):
        count = 0
        for sentence in nested_list:
            if(len(sentence) <= max_len):
                count = count + 1


    max_len = 30
    below_threshold_len(max_len, X_train)

    X_train = pad_sequences(X_train, maxlen=max_len)
    X_test = pad_sequences(X_test, maxlen=max_len)

    path = 'C:/Users/jaehoon/Desktop/00_study/Excel/DATA/best_model.h5'
    PATH = 'C:/Users/jaehoon/Desktop/00_study/Excel/DATA/'
    if os.path.isfile(path):
        print("File exists")
    else:
        embedding_dim = 100
        hidden_units = 128

        model = Sequential()
        model.add(Embedding(vocab_size, embedding_dim))
        model.add(LSTM(hidden_units))
        model.add(Dense(1, activation='sigmoid'))

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)


        mc = ModelCheckpoint(PATH + 'best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

        model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
        history = model.fit(X_train, y_train, epochs=15, callbacks=[es, mc], batch_size=512, validation_split=0.2)

    loaded_model = load_model(PATH + 'best_model.h5')
    #print("\n 테스트 정확도: %.4f" % (loaded_model.evaluate(X_test, y_test)[1]))

    def sentiment_predict(new_sentence):
        new_sentence = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣 ]','', new_sentence)
        new_sentence = okt.morphs(new_sentence, stem=True) # 토큰화
        new_sentence = [word for word in new_sentence if not word in stopwords] # 불용어 제거
        encoded = tokenizer.texts_to_sequences([new_sentence]) # 정수 인코딩

        pad_new = pad_sequences(encoded, maxlen = max_len) # 패딩
        score = float(loaded_model.predict(pad_new)) # 예측

        if(score > 0.5):
          # print(new_sentence)
        #   print(cell)
        #   print("{:.2f}% 확률로 긍정 리뷰입니다.\n".format(score * 100))
          contain.append(cell)
          contain_number.append(score * 100)
        else:
        #   print(cell)
          # print(new_sentence)
        #   print("{:.2f}% 확률로 부정 리뷰입니다.\n".format((1 - score) * 100))
          contain2.append(cell)
          contain2_number.append(score * 100)


    filename = pd.read_excel('C:/Users/jaehoon/Desktop/00_study/Crawling2222/%s.xlsx' %(youtubetitle()))
    sheet = filename['comment']

    for cell in sheet:
        # print(cell)
        output_sentence = str(cell)
        sentiment_predict(output_sentence)

#_----------------------------------------------------------------------

if st.button("검색"):
    con = st.container()
    with st.spinner('크롤링 중입니다.. 기다려주세요♥'):
        Crawling()
    st.success(" YouTube 크롤링이 완료되었습니다!")
    print(youtubetitle())
    openexcel = pd.read_excel('%s.xlsx'%youtubetitle())
    con.write(openexcel)
    
    
if st.button("감정분석"):
    con = st.container()
    with st.spinner('감정분석 중입니다.. 기다려주세요♥'):
        Analysis()
    st.success(" YouTube 크롤링이 완료되었습니다!")
    pd_contain = pd.DataFrame({'긍정댓글' : [contain]})
    pd_contain_number = pd.DataFrame({'확률' : [contain_number]})

    pd_contain2 = pd.DataFrame({'부정댓글' : [contain2]})
    pd_contain2_number = pd.DataFrame({'확률' : [contain2_number]})
    
    result = pd.concat([pd_contain, pd_contain_number], axis=1)
    result1 = pd.DataFrame(result)

    st.header("긍정댓글")
    st.write(result1)

    st.header("부정댓글")
    st.write(pd_contain2)

    #----
    # con = st.container()
    # filename = pd.read_excel('C:/Users/jaehoon/Desktop/00_study/Crawling2222/%s.xlsx' %(youtubetitle()))
    # sheet = filename['comment']

    # for cell in sheet:
    #     output_sentence = str(cell)
    #     Analysis(output_sentence)
    
    # pd_contain = pd.DataFrame(contain)
    # pd_contain2 = pd.DataFrame(contain2)
    # for i, j in zip(contain, contain_number):
    #     st.text(i)
    #     st.text("{:.2f}% 확률로 긍정 리뷰입니다.\n".format(j))
    #     st.text("----------------------------------------------------------------------")

    # for i, j in zip(contain, contain_number):
    #     con.write(i)
    #     con.write("{:.2f}% 확률로 긍정 리뷰입니다.\n".format(j))