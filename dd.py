a,b = map(int, input().split())

reverse_a = int(str(a)[::-1])
reverse_b = int(str(b)[::-1])

if reverse_a > reverse_b:
    print(reverse_a)
else:
    print(reverse_b)