def like():
    try:
        for i in range(1000000):
                if my_str2.find("&list=LL&index=") != -1:
                    mys_url = "".join((my_str2, 's'))
                    lim_str =  mys_url.replace("https://www.youtube.com/watch?v=","")
                    like_del = "&list=LL&index=%ss" % i
                    like_str = lim_str.replace(like_del, "")
                    # find 결과가 false면 -1 리턴. 변환 후엔 당연히 false가 반환
                    if like_str.find("&list=LL&index=")== -1:
                        return like_str
                    else:
                        # 쓰레기 값. if문을 쓰기 위함
                        a=1
        return like_str
    except:
        return my_str3


def zuso():
    try:
        for i in range(13):
                if my_str2.find("&list=WL&index=") != -1:
                    my_url = "".join((my_str2, 's'))
                    m_str =  my_url.replace("https://www.youtube.com/watch?v=","")
                    dele = "&list=WL&index=%ss" % i
                    dele_str = m_str.replace(dele, "")

                    # find 결과가 false면 -1 리턴. 변환 후엔 당연히 false가 반환
                    if dele_str.find("&list=WL&index=")== -1:
                        return dele_str
                    else:
                        # 쓰레기 값. if문을 쓰기 위함
                        a=1
        return dele_str
    except:
        return my_str3

    
def num_re():
    # range(int) -> int는 시간값
    for i in range(10000000):
        # &t= 코드는 고정
        if my_str2.find("&t="):
            # 찾았다면 repplace
            temp ="&t=%ss" %i
            str = my_str2.replace(temp, "")
            
            # find 결과가 false면 -1 리턴. 변환 후엔 당연히 false가 반환
            if str.find("&t=")==-1:
                return str
            else:
                # 쓰레기 값. if문을 쓰기 위함
                a=1
        else:
            # 시간 안적혀 있으면 그대로 리턴
            return str 
    return str

my_str2 = "https://www.youtube.com/watch?v=tokMqkqKXiw&list=WL&index=11"
my_str2 =  my_str2.replace("https://www.youtube.com/watch?v=","")

my_str3 = num_re()
my_str3 = zuso()
my_str3 = like()
print(my_str3)