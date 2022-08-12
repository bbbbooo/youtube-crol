

def zuso():
    try:
        str =  "https://www.youtube.com/watch?v=U3cetm9nnc&list=WL&index=10"
        str = "".join((str, 's'))
        my_str =  str.replace("https://www.youtube.com/watch?v=","")
        for i in range(1000000):
            if my_str.find("&list=WL&index="):
                dele = "&list=WL&index=%ss" % i
                dele_str = my_str.replace(dele, "")

                # find 결과가 false면 -1 리턴. 변환 후엔 당연히 false가 반환
                if dele_str.find("&list=WL&index=")== -1:
                    return dele_str
                else:
                    # 쓰레기 값. if문을 쓰기 위함
                    a=1
            else:
                # 시간 안적혀 있으면 그대로 리턴
                return my_str 
        return dele_str
    except:
        return

zuso()


