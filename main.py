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
from konlpy.tag import Okt
from tqdm import tqdm
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.layers import Embedding, Dense, LSTM
from keras.models import Sequential
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
import pickle


# 웹 제목
st.title("Youtube-CR")


# 주소 입력
input_url = st.text_input(label="URL", value="")
url=input_url
my_str = url.replace("https://www.youtube.com/watch?v=","")


# 주소 뒤 시간이 있다면..
def num_re():
    for i in range(10000000):
        if my_str.find("&t="):
            temp ="&t=%ss" %i
            str = my_str.replace(temp, "")
            
            # find 결과가 false면 -1 리턴됨
            if str.find("&t=")==-1:
                return str
            else:
                # 쓰레기 값
                a=1
        else:
            # 시간 안적혀 있으면 그대로 리턴
            return str 
    return str

# 주소 뒤 시간 제거
my_str2 = num_re()

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
    response = api_obj.commentThreads().list(part='snippet,replies', videoId=my_str2, maxResults=100).execute()
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
    df = df.astype(str)
    df.to_excel('./video_xlxs/%s.xlsx' % (title_get()), header=['comment', 'author', 'date', 'num_likes'], index=None)
    path = './video_xlxs/%s.xlsx' % (title_get())
    while os.path.exists(path) :
        df.to_excel('./video_xlxs/%s.xlsx' % (title_get()), header=['comment', 'author', 'date', 'num_likes'], index=None)
        break
    
    
    
contain = []        #긍정 cell
contain_number =[]  #긍정 확률
contain2 = []       #부정 cell
contain2_number = []#부정 확률

def Analysis():
    tokenizer = Tokenizer()
    okt = Okt()
    max_len = 30

    # 불용어
    stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']
    #PATH = '/Users/82102/Desktop/project/yt_cr/model/save_model/'
    PATH = '/Users/82102/Desktop/project/yt_cr/model_test/'
    #PATH2 = '/Users/82102/Desktop/project/yt_cr/model/token/'
    PATH2 = '/Users/82102/Desktop/project/yt_cr/model_test/'
    
    #모델 및 토큰 불러오기
    model = load_model(PATH + 'best_model.h5')
    with open(PATH2+'tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)


    # 감정 예측
    def sentiment_predict(new_sentence):
        new_sentence = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣 ]','', str(new_sentence))
        new_sentence = okt.morphs(new_sentence, stem=True) # 토큰화
        new_sentence = [word for word in new_sentence if not word in stopwords] # 불용어 제거
        encoded = tokenizer.texts_to_sequences([new_sentence]) # 정수 인코딩
        pad_new = pad_sequences(encoded, maxlen = max_len) # 패딩
        score = float(model.predict(pad_new)) # 예측
        
        # 긍정적이라면 contain 리스트에 추가
        if(score > 0.5):
            contain.append(list)
            contain_number.append(score * 100)
        # 부정적이라면 contain2 리스트에 추가
        else:
            contain2.append(list)
            contain2_number.append( (1 - score) * 100)


    # 다른 함수에서도 쓰기 위해 global(전역변수) 선언
    global filename, sheet
    filename = pd.read_excel('/Users/82102/Desktop/project/yt_cr/video_xlxs/%s.xlsx' % title_get())
    sheet = filename['comment']


    # comment 칼럼의 각각의 데이터를 읽기
    for cell in sheet:
        list = []
        output_sentence = str(cell)

        # 댓글에 html 코드가 존재한다면...
        if "</a>" in output_sentence:
            split = output_sentence.split('</a>')

            # 지우고 나서 split의 1번째 index에 '' <<(이름 모를 공백값, NULL값 아님)이 있다면 제거. 출력 안함
            if split[1] == '':
                continue
            # 없으면 list에 저장
            else:
                split2 = split[1].replace('<br>', '')
                list.append(split2)

        # 아무것도 해당 안되면 바로 list에 추가
        else:
            list.append(output_sentence)

        # 감정 예측
        sentiment_predict(list)

# 원형 차트 생성
def Create_plot():
    
  # 1. 저장된 엑셀 파일의 comments 길이를 계산
  # 2. 긍정, 부정으로 저장된 comments의 길이를 계산, 이를 활용해 비율을 계산
  # 3. 이를 토대로 원형 차트 출력
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
  
  st.pyplot(fig)

#-----------------------------------------------------------------------------------------------------------------------------------------------------------

# 다른거 눌렀을때
def etc():
    st.write('not yet')

#-----------------------------------------------------------------------------------------------------------------------------------------------------------


# 현재 선택한 옵션
option = st.sidebar.selectbox(
    '원하는 기능을 선택해주세요',
    ['유튜브 댓글 분석', 'etc'])

'현재 선택하신 옵션은 : ', option

#------------------------------------------------------------------------------------------------------------


# 댓글 분석 눌렀을때...
def Youtube_Comments_Analysis():
    # Search 버튼 클릭 시....
    if st.button("검색"):
        con = st.container()
        with st.spinner("검색중...."):
            time.sleep(2)
        st.success("검색이 완료됐습니다. 감정 분석까지 약 1분이 소요됩니다.")

        # 결과(입력 주소) 출력
        con.caption("검색 결과")
        con.write(f"입력하신 주소는 {str(input_url)} 입니다.")
        # 썸네일 출력
        st.header('썸네일')
        st.image(get_thumbnail(my_str2))

        # 댓글 크롤링
        Crawling()
            
        # 데이터 분석
        Analysis()
        
        # 긍정 댓글, 확률
        global pd_contain, pd_contain2
        pd_contain = pd.DataFrame({'긍정 댓글' : contain})
        pd_contain_number = pd.DataFrame({'확률': contain_number})
        pos_result = pd.concat([pd_contain, pd_contain_number], axis=1)

        #부정 댓글, 확률
        pd_contain2 = pd.DataFrame({'부정 댓글' : contain2})
        pd_contain_number2 = pd.DataFrame({'확률': contain2_number})
        neg_result = pd.concat([pd_contain2, pd_contain_number2], axis=1)

        # 데이터 프레임 출력
        st.header("긍정")
        st.write(pos_result)

        st.header("부정")
        st.write(neg_result)

        # 원형 차트 출력
        st.header('원형 차트')
        Create_plot()

# 유튜브 댓글 분석 선택하면 해당 기능 실행...
if option == '유튜브 댓글 분석':
    Youtube_Comments_Analysis()

# 추가 기능
if option == 'etc':
    st.write('not yet')