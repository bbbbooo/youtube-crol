from asyncio.windows_events import NULL
import base64
from cmath import pi
from dataclasses import replace
from datetime import datetime
from email.mime import image
from re import L, S
import sqlite3 as sq
import os
from matplotlib.pyplot import table
import pandas as pd
from PIL import Image
from sqlalchemy import null
import streamlit as st
import socket
from PIL import Image
import matplotlib.pyplot as plt 
from io import BytesIO

st.header('ㅁㄴㅇㄴㅁ')

name = '큰일났다 카트라이더'

# filename = pd.read_excel('/Users/82102/Desktop/project/yt_cr/video_xlsx/%s.xlsx' % name)



        

def save_db():
    # db 생성
        conn = sq.connect('test.db', isolation_level=None, )
        print('db 연동에 성공했습니다.')

        # 커서 획득
        c = conn.cursor()
        print('커서 획득에 성공했습니다.')

        # 경로 지정 및 파일 오픈
        pex = './result_video/%s_positive.xlsx' % name
        nex = './result_video/%s_negative.xlsx' % name
        cpath = './result_image/%s_chart.png' % name
        ppath = './result_wc/%s_positive.png' % name
        npath = './result_wc/%s_negative.png' % name
        
        print('파일 반환에 성공했습니다.')

        # id 테이블 저장
        c.execute('CREATE TABLE IF NOT EXISTS ipList \
            (id integer primary key AUTOINCREMENT, vid text);')
        c.execute('INSERT INTO ipList(vid) VALUES (?);', (name,))
        print('IP주소가 할당되었습니다.')

        # data 테이블 저장
        c.execute("CREATE TABLE IF NOT EXISTS edata \
            (id integer primary key AUTOINCREMENT, vid text , pex text, nex text, chart text, pwc text, nwc text);")
        print('테이블이 생성되었습니다.')

        c.execute('INSERT INTO edata(vid, pex, nex, chart, pwc, nwc) VALUES (?,?,?,?,?,?);', (name, pex, nex , cpath, ppath, npath))
        print('데이터를 저장했습니다.')# 테이블 생성
        
        
        
        # ipList 테이블에서 id값 가져오기
        c.execute('SELECT id FROM ipList;')
        all_id = c.fetchall()
        
        def search_history():
            for row in all_id:
                # idList 테이블의 id값에 접근, id값이 5 이상일 경우 초기화
                # id값은 검색 기록 버튼을 할당하기 위한 값. 
                print(row)
                int_row = int(''.join(map(str, row)))
                     
                if int_row == 1:
                    if st.sidebar.button('1', key=0):
                        c.execute('SELECT pex, nex, chart, pwc, nwc FROM edata WHERE id == 1;')

                        # 테이블의 0,1....4번째 index 값 가져오기
                        all = c.fetchone()

                        pex = all[0]
                        nex = all[1]
                        chart = all[2]
                        pwc = all[3]
                        nwc = all[4]
                        print("데이터 베이스에서 경로를 가져오는데 성공했습니다.")

                        # 오픈
                        open_pex = pd.read_excel(pex)
                        open_nex = pd.read_excel(nex)
                        open_chart =  Image.open(chart)
                        open_pwc =  Image.open(pwc)
                        open_nwc =  Image.open(nwc)


                        # 출력
                        st.write(open_pex)
                        st.write(open_nex)
                        st.image(open_chart)
                        st.image(open_pwc)
                        st.image(open_nwc)
                    
                if int_row == 2:
                    if st.sidebar.button('2', key=1):
                        c.execute('SELECT pex, nex, chart, pwc, nwc FROM edata WHERE id == 2;')

                        # 테이블의 0,1....4번째 index 값 가져오기
                        all = c.fetchone()

                        pex = all[0]
                        nex = all[1]
                        chart = all[2]
                        pwc = all[3]
                        nwc = all[4]
                        print("데이터 베이스에서 경로를 가져오는데 성공했습니다.")

                        # 오픈
                        open_pex = pd.read_excel(pex)
                        open_nex = pd.read_excel(nex)
                        open_chart =  Image.open(chart)
                        open_pwc =  Image.open(pwc)
                        open_nwc =  Image.open(nwc)


                        # 출력
                        st.write(open_pex)
                        st.write(open_nex)
                        st.image(open_chart)
                        st.image(open_pwc)
                        st.image(open_nwc)
                        
                if int_row == 3:
                    if st.sidebar.button('3', key=2):
                        c.execute('SELECT pex, nex, chart, pwc, nwc FROM edata WHERE id == 3;')

                        # 테이블의 0,1....4번째 index 값 가져오기
                        all = c.fetchone()

                        pex = all[0]
                        nex = all[1]
                        chart = all[2]
                        pwc = all[3]
                        nwc = all[4]
                        print("데이터 베이스에서 경로를 가져오는데 성공했습니다.")

                        # 오픈
                        open_pex = pd.read_excel(pex)
                        open_nex = pd.read_excel(nex)
                        open_chart =  Image.open(chart)
                        open_pwc =  Image.open(pwc)
                        open_nwc =  Image.open(nwc)


                        # 출력
                        st.write(open_pex)
                        st.write(open_nex)
                        st.image(open_chart)
                        st.image(open_pwc)
                        st.image(open_nwc)
                    
                if int_row > 3:
                    c.execute('DROP TABLE ipList;')
                    c.execute('DROP TABLE edata;')
                    print("데이터 삭제 완료")
                    st.sidebar.write("검색 기록이 초기화 됐습니다.")
                    conn.commit()
                         
        
        search_history()
        
        conn.commit()
        conn.close()


save_db()
