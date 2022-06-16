from selenium import webdriver
import time
from openpyxl import Workbook
import pandas as pd
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
from sqlalchemy import null
from googleapiclient.discovery import build
import os
import re
import streamlit as st
import pafy as pa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
from konlpy.tag import Okt
from tqdm import tqdm
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.layers import Embedding, Dense, LSTM
from keras.models import Sequential
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint





# 웹 제목
st.title("Youtube-CR")

#주소 가져오기 및 videoid 추출
input_url = st.text_input(label="URL", value="")
url=input_url
my_str = url.replace("https://www.youtube.com/watch?v=","")

#썸네일 출력
def get_thumbnail(url):
    id = url
    img = 'https://img.youtube.com/vi/{}/0.jpg'.format(id)
    return img    

# 제목 가져와서 변환 후 변환 값 return
def title_get():
    videoinfo = pa.new(url)
    video_title = videoinfo.title
    
    #제목 특수기호 있으면 공백으로 치환
    rp_video_title = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…《\》]', '', video_title)
    return rp_video_title


# 댓글 크롤링하여 xlsx 형태로 video_xlxs 폴더에 저장
def Crawling():
    #api키 입력
    api_key = 'AIzaSyDCLqtKIMyBZ82hWpUj1QcTg_glkAlk1kk'
    comments = list()
    api_obj = build('youtube', 'v3', developerKey=api_key)
    response = api_obj.commentThreads().list(part='snippet,replies', videoId=my_str, maxResults=100).execute()
    while response:
        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']
            comments.append([comment['textDisplay'], comment['authorDisplayName'], comment['publishedAt'], comment['likeCount']])
            # 대댓글 불러오기
            # if item['snippet']['totalReplyCount'] > 0:
            #     for reply_item in item['replies']['comments']:
            #     d    reply = reply_item['snippet']
            #         comments.append([reply['textDisplay'], reply['authorDisplayName'], reply['publishedAt'], reply['likeCount']])
        if 'nextPageToken' in response:
            response = api_obj.commentThreads().list(part='snippet,replies', videoId='sWC-pp6CXpA', pageToken=response['nextPageToken'], maxResults=100).execute()
        else:
            break
    df = pd.DataFrame(comments)
    df.to_excel('./video_xlxs/%s.xlsx' % (title_get()), header=['comment', 'author', 'date', 'num_likes'], index=None)
    path = './video_xlxs/%s.xlsx' % (title_get())
    while os.path.exists(path) :
        df.to_excel('./video_xlxs/%s.xlsx' % (title_get()), header=['comment', 'author', 'date', 'num_likes'], index=None)
        break



##################### 데이터 전처리


# 데이터셋 다운로드. 완료하였다면 주석처리 할 것.
#urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt", filename="ratings_train.txt")
#urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt", filename="ratings_test.txt")


# 긍정, 부정 문장을 담는 배열
contain = []        #긍정 cell
contain_number =[]  #긍정 확률
contain2 = []       #부정 cell
contain2_number = []#부정 확률

def Analysis():
    train_data = pd.read_table('/Users/82102/Desktop/project/yt_cr/study_analy/rating_data/ratings_train.txt')
    test_data = pd.read_table('/Users/82102/Desktop/project/yt_cr/study_analy/rating_data/ratings_test.txt')

    # document 열과 label 열의 중복을 제외한 값의 개수
    #train_data['document'].nunique(), train_data['label'].nunique()

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


    path = '/Users/82102/Desktop/project/yt_cr/study_analy/model/best_model.h5'
    PATH = '/Users/82102/Desktop/project/yt_cr/study_analy/model/'
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
            new_sentence = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣 ]','', str(new_sentence))
            new_sentence = okt.morphs(new_sentence, stem=True) # 토큰화
            new_sentence = [word for word in new_sentence if not word in stopwords] # 불용어 제거
            encoded = tokenizer.texts_to_sequences([new_sentence]) # 정수 인코딩
            pad_new = pad_sequences(encoded, maxlen = max_len) # 패딩
            score = float(loaded_model.predict(pad_new)) # 예측

            if(score > 0.5):
                contain.append(cell)
                contain_number.append(score * 100)

            else:
                contain2.append(cell)
                contain2_number.append( (1 - score) * 100)


    global filename, sheet
    filename = pd.read_excel('/Users/82102/Desktop/project/yt_cr/video_xlxs/%s.xlsx' % title_get())
    sheet = filename['comment']
    #pw = pd.DataFrame(list(filename.items()), columns=['comment', 'author'])


    #sheet.replace("&lt;a href=https://www.youtube.com/watch?v=kR7qz8liQqA&amp;amp;t=7m57s&gt;7:57&lt;/a&gt;", "")


    # comment 칼럼의 각각의 데이터를 읽기
    for cell in sheet:
        output_sentence = str(cell)
        sentiment_predict(output_sentence)


# 원형 그래프 생성
def Create_plot():
        allen = len(sheet)
        poslen = len(pd_contain)
        neglen = len(pd_contain2)

        pos_ratio = (poslen/allen) * 100
        neg_ratio = (neglen/allen) * 100

        labels = ['Positive', 'Negative']
        ratio = pos_ratio, neg_ratio

        fig, ax = plt.subplots()
        ax.pie(ratio, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
        ax.axis('equal')

        st.pyplot(fig, clear_figure=True)
        #plt.savefig("mygraph.png")


# Search 버튼 클릭 시....
if st.button("Search"):
    con = st.container()
    with st.spinner("Searching...."):
        time.sleep(2)
    st.success("The search was successful. It takes approximately one minute to analyze the results. Just a moment, please.")
    
    # 결과(입력 주소) 출력
    con.caption("Result")
    con.write(f"The entered video address is {str(input_url)}")
    # 썸네일 출력
    st.header('Thumbnail')
    st.image(get_thumbnail(my_str))
    
    # 댓글 크롤링
    Crawling()
    
    # 데이터 분석
    Analysis()
    
    # 긍정 댓글, 확률
    pd_contain = pd.DataFrame({'Postive_Comments' : contain})
    pd_contain_number = pd.DataFrame({'Probability': contain_number})
    pos_result = pd.concat([pd_contain, pd_contain_number], axis=1)
    
    #부정 댓글, 확률
    pd_contain2 = pd.DataFrame({'Negative' : contain2})
    pd_contain_number2 = pd.DataFrame({'Probability': contain2_number})
    neg_result = pd.concat([pd_contain2, pd_contain_number2], axis=1)
    
    # 출력
    st.header("Positive")
    st.write(pos_result)
    
    st.header("Negative")
    st.write(neg_result)
    
    # pie plot
    st.header('Pie Plot')
    Create_plot()
    
    

        