# Python TIL
TrendMiner쪽 개발을 하게 되면서 오랫만에 python으로 끄적이게 되었다. 하면서 썼던 내용들을 추가하려고 한다.
썼던 내용들은 크게 어려운 내용은 아니지만 정리 차원에서 적어둔다.

## Read File
```python
# 파일 읽기
f = open('taglist.txt','r',enconding='utf-8')
# 파일에 붙은 \n 제거
line = f.read().splitlines()
``` 

## String
```python
# 문자열에 특정 문자가 있는지 확인
test = "This is test"
if "This" in test:
    print(test)

```

# Pandas
```python
# Series Data 
# ds.index = pandas
# type(ds.values) = numpy.ndarray
#series to DF
df = ds.to_frame()

#DF to SQLite
df.to_sql('val',conn)
```

# SQLite
```python
import sqlite3

# Set Database
conn = sqlite3.connect('test.db')
# Pandas DF to SQLite
df.to_sql('val',conn)
# If table exists / append , replace , fail
df.to_sql('val',conn,if_exists="append")

conn.close()
```