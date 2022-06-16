#주소 가져오기 및 videoid 추출
# def replace_num():
#     for i in range(1, 101):    # 1부터 100까지 100번 반복
#         num_arr = i
#     return num_arr


input_url = 'https://www.youtube.com/watch?v=kR7qz8liQqA&t=1s'
url=input_url
my_str = url.replace("https://www.youtube.com/watch?v=","")


def num_re():
    for i in range(3):
        if my_str.find("&t="):
            temp ="&t=%ss" %i
            str = my_str.replace(temp, "")
            
            # find 결과가 false면 -1 리턴됨
            if str.find("&t=")==-1:
                return str
            else:
                a=1
        else:
            return str 
    return str

my_str2 = num_re()


print(my_str2)