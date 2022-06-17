#주소 가져오기 및 videoid 추출
# def replace_num():
#     for i in range(1, 101):    # 1부터 100까지 100번 반복
#         num_arr = i
#     return num_arr

 #pw = pd.DataFrame(list(filename.items()), columns=['comment', 'author'])
    #sheet.replace("&lt;a href=https://www.youtube.com/watch?v=kR7qz8liQqA&amp;amp;t=7m57s&gt;7:57&lt;/a&gt;", "")

from string import digits
import pandas as pd
import re

from zmq import NULL

filename = pd.read_excel('/Users/82102/Desktop/project/yt_cr/video_xlxs/벨베스 사기네.xlsx')
sheet = filename['comment']

def Sub_comments():
    list = []
    
    for cell in sheet:
        #cmrp = re.sub('[^가-힣]', '', str(cell))
        if "</a>" in cell:
            split = cell.split('</a>')
            if split[1] == '':
                continue
            else:
                list.append(split[1])
        
        else:
            list.append(cell)
            
    return list

    
        
        

print(Sub_comments())



# def num_re():
#     for i in range(1000000):
#         if my_str.find("&t="):
#             temp ="&t=%ss" %i
#             str = my_str.replace(temp, "")
            
#             # find 결과가 false면 -1 리턴됨
#             if str.find("&t=")==-1:
#                 return str
#             else:
#                 a=1
#         else:
#             return str 
#     return str
