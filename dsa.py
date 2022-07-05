from cmath import pi
from datetime import datetime
from email.mime import image
import sqlite3 as sq
import os
import pandas as pd
from PIL import Image

name = '모두가 속은 유미의 릭트쇼'

# filename = pd.read_excel('/Users/82102/Desktop/project/yt_cr/video_xlsx/%s.xlsx' % name)

time = datetime.now()
now = time.strftime('%Y-%m-%d %H:%M:%S')

# 파일 오픈
def openfile(filename):
    with open(filename, 'rb') as file:
        files = file.read()
    return files


# db 생성
try:
    conn = sq.connect('test.db', isolation_level=None)
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

    pos_ex = openfile(pex)
    neg_ex = openfile(nex)
    chart = openfile(cpath)
    pwc = openfile(ppath)
    nwc = openfile(npath)
    print('파일 반환에 성공했습니다.')
    
    # 테이블 생성
    c.execute("CREATE TABLE IF NOT EXISTS table1 \
        (id integer PRIMARY KEY AUTOINCREMENT, pex BLOB, nex blob, tdate text DEFAULT(datetime('now','localtime')), chart BLOB, pwc blob, nwc blob);")
    print('테이블이 생성되었습니다.')

    c.execute('INSERT INTO table1(pex, nex, tdate, chart, pwc, nwc) VALUES (?,?,?,?,?,?);', (pos_ex, neg_ex, now, chart, pwc, nwc))
    print('데이터를 저장했습니다.')
    # 데이터 정보 출력
    # c.execute('SELECT * from table1')
    # all = c.fetchall()
    # print(all)

    conn.commit()
    conn.close()

except:
    print('db 연동에 실패했습니다.')
    
finally:
    if conn:
        if conn.close():
            print('db 연결이 닫혔습니다.')