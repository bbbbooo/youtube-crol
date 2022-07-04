from base64 import encode
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
from wordcloud import WordCloud
from collections import Counter
from PIL import Image
from konlpy.tag import Okt
from tqdm import tqdm
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.layers import Embedding, Dense, LSTM
from keras.models import Sequential
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
import pickle
from PIL import Image

#####

# 해당 코드는 main.py의 url 입력, 크롤링 부분 제외한 테스트 코드
# 따라서 현 파일의 83줄 read.excel에서 main.py을 통해 생성된 엑셀 파일의 이름을 직접 적어야 함
# 실행은 streamlit run test.py

#####


st.header('asdsad')

tokenizer = Tokenizer()
okt = Okt()

name = '잘나가는 고깃집'
ppath = './result_video/%s_positive.xlsx' % name
npath = './result_video/%s_negative.xlsx' % name





#######
# 1. 다른사람한테 받은 모델과 토큰값은 model_test에 저장
# 1-1. 이 경우 아래 PATH의 model_test 주석을 해제

# 2. model 폴더에서 모델과 토큰값을 생성한 경우
# 2-1. 아래 PATH의 save_model 주석을 해제
#######


if not os.path.exists(ppath):
    def aa():
        PATH = '/Users/82102/Desktop/project/yt_cr/model/save_model/'
        #PATH = '/Users/82102/Desktop/project/yt_cr/model_test/'
        loaded_model = load_model(PATH + 'best_model.h5')

        PATH2 = '/Users/82102/Desktop/project/yt_cr/model/token/'
        #PATH2 = '/Users/82102/Desktop/project/yt_cr/model_test/'
        with open(PATH2+'tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)

        stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']
        max_len = 30


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



        filename = pd.read_excel('/Users/82102/Desktop/project/yt_cr/video_xlsx/%s.xlsx' % name)
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


        #긍정
        pd_contain = pd.DataFrame({'Postive' : contain})
        pd_contain_number = pd.DataFrame({'postive-comments': contain_number})
        result = pd.concat([pd_contain, pd_contain_number], axis=1)

        #부정
        pd_contain2 = pd.DataFrame({'Negative' : contain2})
        pd_contain_number2 = pd.DataFrame({'negative-comments': contain2_number})
        result2 = pd.concat([pd_contain2, pd_contain_number2], axis=1)

        # 결과 저장
        result.to_excel('./result_video/%s_positive.xlsx' % name, header=['comments', 'Probability'])
        result2.to_excel('./result_video/%s_negative.xlsx' % name, header=['comments', 'Probability'])


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
          plt.savefig('./result_image/%s_chart.png' % name)


          st.pyplot(fig)

        def dsadsa(list_dsa):
            data_list = []
            for i in list_dsa:
                data_list.append(i)
            return str(data_list)

        # 긍정 워드 클라우드
        def Create_pword():
            okt = Okt()

            pos = dsadsa(contain)

            pn = okt.nouns(pos)
            pw = [n for n in pn if len(n) > 1]
            pc = Counter(pw)
            pwc = WordCloud(font_path='malgun', width=400, height=400, scale=2.0, max_font_size=250)

            pg = pwc.generate_from_frequencies(pc)
            pfig = plt.figure()

            plt.imshow(pg, interpolation='bilinear')
            plt.axis('off')
            plt.savefig('./result_wc/%s_positive.png' % name)
            plt.show()

            st.markdown('긍정')
            st.pyplot(pfig)

        # 부정 워드 클라우드
        def Create_nword():
            okt = Okt()
            neg = dsadsa(contain2)

            nn = okt.nouns(neg)
            nw = [n for n in nn if len(n) > 1]
            nc = Counter(nw)
            nwc = WordCloud(font_path='malgun', width=400, height=400, scale=2.0, max_font_size=250)

            ng = nwc.generate_from_frequencies(nc)
            nfig = plt.figure()

            plt.imshow(ng, interpolation='bilinear')
            plt.axis('off')
            plt.savefig('./result_wc/%s_negative.png' % name)
            plt.show()

            st.markdown('부정')
            st.pyplot(nfig)

        st.header('Word Cloud')
        Create_pword()
        Create_nword()

        # pie plot
        st.header('Pie Plot')
        Create_plot()
    aa()
    
else:
    path = './result_image/'
    path1 = './result_wc/'
    def load_chart():
        chart = Image.open(path + '%s_chart.png' % name)
        return chart
    
    def load_pwc():
        pwc = Image.open(path1 + '%s_positive.png' % name)
        return pwc
    
    def load_nwc():
        nwc = Image.open(path1 + '%s_negative.png' % name)
        return nwc
    
    rep = pd.read_excel('./result_video/%s_positive.xlsx' % name)
    ren = pd.read_excel('./result_video/%s_negative.xlsx' % name)
    if st.sidebar.button('%s' % name):
        st.write(rep)
        st.write(ren)
        chart = load_chart()
        pwc = load_pwc()
        nwc = load_nwc()

        st.image(chart)
        st.image(pwc)
        st.image(nwc)
        
        st.stop()

