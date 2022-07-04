import sqlite3 as sq

# db 생성
conn = sq.connect('test.db', isolation_level=None)

# 커서 획득
c = conn.cursor()

# 테이블 생성
c.execute("CREATE TABLE IF NOT EXISTS table1 \
    (id integer PRIMARY KEY, date text)")