from asyncio.windows_events import NULL
import base64
from cmath import pi
from dataclasses import replace
from datetime import datetime
from email.mime import image
from re import L
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


# ip 가져오기
def get_ip():
    ip = socket.gethostbyname(socket.getfqdn())
    return ip



        

def save_db():
    # db 생성
    try:
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
            (id integer primary key AUTOINCREMENT, ip text, vid text);')
        c.execute('INSERT INTO ipList(ip, vid) VALUES (?,?);', (get_ip(), name))
        print('IP주소가 할당되었습니다.')

        # data 테이블 저장
        c.execute("CREATE TABLE IF NOT EXISTS edata \
            (id integer primary key AUTOINCREMENT, vid text , pex text, nex text, chart text, pwc text, nwc text);")
        print('테이블이 생성되었습니다.')

        c.execute('INSERT INTO edata(vid, pex, nex, chart, pwc, nwc) VALUES (?,?,?,?,?,?);', (name, pex, nex , cpath, ppath, npath))
        print('데이터를 저장했습니다.')# 테이블 생성

        def vid_search():
            for row in c.execute('SELECT id FROM ipList;'):
                # idList 테이블의 id값에 접근, id값이 5 이상일 경우 초기화
                # id값은 검색 기록 버튼을 할당하기 위한 값. 
                int_row = int(''.join(map(str, row)))
                # int_row2 = int(''.join(map(str, row2)))
                if int_row == 1:
                    st.sidebar.button('1', key=0)
                    #c.execute('SELECT vid FROM edata;')
                    #st.write(name)
                    print('데이터 출력')
                if int_row == 2:
                    st.sidebar.button('2', key=1)
                    print('데이터 출력')
                if int_row == 3:
                    st.sidebar.button('3', key=2)
                    print('데이터 출력')
                if int_row == 4:
                    st.sidebar.button('4', key=3)
                    print('데이터 출력')
                if int_row == 5:
                    st.sidebar.button('5', key=4)
                    print('데이터 출력')
                if int_row > 5:
                    c.execute('DROP TABLE ipList;')
                    c.execute('DROP TABLE edata;')
                    print("데이터 삭제 완료")
                    st.sidebar.write("검색 기록이 초기화 됐습니다.")
                    conn.commit()
            
        
        
        def Search_history():
            if c.execute('SELECT id FROM ipList WHERE id = 1;'):
                
                c.execute('SELECT pex, nex, chart, pwc, nwc FROM edata WHERE id = 1;')

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
                
                # if st.button('1', key=1):
                #     # id값이 1인 edata 테이블의 정보 가져오기
                #     c.execute('SELECT pex, nex, chart, pwc, nwc FROM edata WHERE id = 1;')

                #     # 테이블의 0,1....4번째 index 값 가져오기
                #     all = c.fetchone()

                #     pex = all[0]
                #     nex = all[1]
                #     chart = all[2]
                #     pwc = all[3]
                #     nwc = all[4]
                #     print("데이터 베이스에서 경로를 가져오는데 성공했습니다.")

                #     # 오픈
                #     open_pex = pd.read_excel(pex)
                #     open_nex = pd.read_excel(nex)
                #     open_chart =  Image.open(chart)
                #     open_pwc =  Image.open(pwc)
                #     open_nwc =  Image.open(nwc)


                #     # 출력
                #     st.write(open_pex)
                #     st.write(open_nex)
                #     st.image(open_chart)
                #     st.image(open_pwc)
                #     st.image(open_nwc)
                    
            # if c.execute('SELECT id FROM ipList WHERE id = 2;'):
            #     if st.button('2', key = 2):
            #         st.write("출력")
                    
            # if c.execute('SELECT id FROM ipList WHERE id > 5;'):
            #     c.execute('DROP TABLE ipList;')
            #     c.execute('DROP TABLE edata;')
            #     print("데이터 삭제 완료")
            #     st.sidebar.write("검색 기록이 초기화 됐습니다.")

            
        Search_history()

        conn.commit()
        conn.close()

    except Exception as e:
        print('db 연동에 실패했습니다.')
        print(e)

    finally:
        if conn:
            if conn.close():
                print('db 연결이 닫혔습니다.')
                
save_db()