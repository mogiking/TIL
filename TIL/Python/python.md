# Python TIL
TrendMiner쪽 개발을 하게 되면서 오랫만에 python으로 끄적이게 되었다. 하면서 썼던 내용들을 추가하려고 한다.
썼던 내용들은 크게 어려운 내용은 아니지만 정리 차원에서 적어둔다.

# Basic

## 자료형

### Number수 자료형
수 자료형은 정수, 실수로 구분 가능하다.

#### Integer(int) 정수형
정수는 0,양의 정수, 음의 정수로 나뉜다.
초기화는 다음과 같이 할 수 있다.
```python
positive_int = 1
negative_int = -1
zero = 0
```

#### Real number(decimal,float, double) 실수형
이진 연산에서 컴퓨터는 소수점을 정확하게 나타내기 힘듭니다. 때문에 부동소수점을 이용하여 연산하게 되는데, 부동소수점은 0.1 , 0.01 등을 정확한 수로 나타낼 수 없기 때문에 정확도 문제가 발생한다.
실수는 다음과 같이 표현할 수 있다.

```python
positive_real = 1.123
negative_real = -1.123
zero_real = .1
real_zero = 1.

real=zero_real*zero_real
print(real)
# 0.010000000000000002

# 지수 연산
a = 1.23e5
print(a)
# 123000.0
a = 4.32e-4
print(a)
# 0.000432
```
위의 코드를 실행하면 0.1 * 0.1 이 0.01 이 아닌 0.010000000000000002 과 같은 숫자가 나오는 것을 확인할 수 있다.
따라서 비교 연산시에 round() 함수를 사용하여 비교를 수행한다.
round(a,b) 에서 a = 반올림하고자 하는 인자, b는 출력 소수점 자리수 이다.
```python
decimal = 1.126

print(round(decimal,1)) 
# 1.1
print(round(decimal,2)) 
# 1.13
print(round(decimal,3)) 
# 1.126
```

#### 연산
기본적인 사칙연산 (+,-,*,/) 가 가능하다. 나누기(/)의 경우, 결과를 기본적으로 실수형으로 처리한다.
외에도 나머지 연산자(%), 몫 연산자(//), 거듭제곱 연산자(**) 등이 가능하다.

### 리스트 자료형

#### 리스트 생성
```python
empty_list = list()
empty_list = []
list_sample= [1,2,3,4,5,6,7]

print(list_sample[0]) 
# 1 ; 제일 앞
print(list_sample[-1]) 
# 7 ; 뒤에서 첫번째
print(list_sample[-5]) 
# 3 ; 뒤에서 다섯번째
```


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