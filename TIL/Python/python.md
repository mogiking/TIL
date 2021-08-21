# Python TIL
개인적으로 썼던 함수, 혹은 Tip들을 기록하기 위해 해당 문서를 작성한다.


---
# Basic
이 섹션은 Book: **이것이 코딩 테스트다 with 파이썬**.

**Appendix A\[코딩테스트를 위한 파이썬 문법\]** 을 기반으로 한다.

- Python은 'tab'과 'space'에 민감하다. 표준은 tab 1회 당 space 4회로 취급한다. 따라서 이 문서에서도 이 표준을 따라간다.
    - tab과 space에서 주석의 레벨도 민감한 경우가 있다.
    - e.g. tab이 없는 곳에서 Multiline 주석을 시작했으면 같은 레벨에서 주석을 닫아야한다.
- Python은 자료형을 명시하지 않는다.
- Python은 세미콜론(;)을 사용하지 않는다.
- 주석은 #과 ''' -multi lines- ''' 로 표시한다.


## 자료형
명시적으로 자료형을 선언해야하는 C, C++, Java 등과 다르게 파이썬은 자동으로 자료형이 매핑된다. 때문에 명시적으로 형 변환을 해야하는 경우는 casting 함수를 쓴다.
```python
a = 15
float_a = float(a) 
# > 15.0
```

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
# > 0.010000000000000002

# 지수 연산
a = 1.23e5
print(a)
# > 123000.0
a = 4.32e-4
print(a)
# > 0.000432
```
위의 코드를 실행하면 0.1 * 0.1 이 0.01 이 아닌 0.010000000000000002 과 같은 숫자가 나오는 것을 확인할 수 있다.
따라서 비교 연산시에 round() 함수를 사용하여 비교를 수행한다.
round(a,b) 에서 a = 반올림하고자 하는 인자, b는 출력 소수점 자리수 이다.
```python
decimal = 1.126

print(round(decimal,1)) 
# > 1.1
print(round(decimal,2)) 
# > 1.13
print(round(decimal,3)) 
# > 1.126
```

#### 연산
기본적인 사칙연산 (+,-,\*,/) 가 가능하다. 나누기(/)의 경우, 결과를 기본적으로 실수형으로 처리한다.
외에도 나머지 연산자(%), 몫 연산자(//), 거듭제곱 연산자(\*\*) 등이 가능하다.


### 리스트 자료형
다른 언어와 같이 Array기능을 지원하고 있고, 내부적으로는 연결 리스트 자료구조를 채택하고 있어 'append, remove, insert' 등의 메서드를 지원한다.
리스트는 인덱스(주소의 개념)와 데이터(원소)로 구성된다. 인덱스는 0부터 시작한다.
#### 리스트 생성 & 출력
```python
# list 초기화
empty_list = list()
empty_list = []
list_sample= [1,2,3,4,5,6,7]

# 리스트 출력
print(list_sample)
# > [1, 2, 3, 4, 5, 6, 7]
print(list_sample[0]) 
# > 1 ; 제일 앞
print(list_sample[-1]) 
# > 7 ; 뒤에서 첫번째
print(list_sample[-5]) 
# > 3 ; 뒤에서 다섯번째
```

#### 리스트 슬라이싱
파이썬에서는 구역을 정해서 리스트를 지정할 수 있다. 예를 들어 0번 인덱스부터 4번 인덱스까지, 뒤에서 7번째 인덱스부터 뒤에서 3번째까지 등등 인덱스의 범위를 지정해서 출력이 가능하다.
슬라이싱을 통해 출력되는 결과는 리스트 형을 반환(return)한다.
```python
# list 초기화
list_sample= [1,2,3,4,5,6,7]

print(list_sample)
# > [1, 2, 3, 4, 5, 6, 7]
print(list_sample[1:6]) 
# > [2, 3, 4, 5, 6] ; 1번 인덱스([1])부터 6번 인덱스([7]) '이전'까지
print(list_sample[-7:-1]) 
# > [1, 2, 3, 4, 5, 6] ; 뒤에서 7번째 인덱스부터 뒤에서 첫번째 인덱스 '이전'까지
```

#### 리스트 컴프리헨션
리스트를 초기화 하는 방법 중의 하나. 리스트 초기화 시에 조건문, 반복문을 넣어서 리스트를 초기화 할 수 있다.
```python
# 0 ~ 20 까지의 제곱을 반환하는 리스트
list_square = [i * i for i in range(21)]
print(list_square)
'''
> [0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225, 256, 289, 324, 361, 400]
'''

# 1차원(N) 0 배열 생성
n = 10
list_n = [0] * n
print(list_n)
'''
> [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
'''

# 2차원(NxM) 0 배열 생성 
n = 3
m = 4
list_nm = [ [0] * m for i in range(n)]
'''
> [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
'''
```

### 문자열
파이썬의 문자열은 내부적으로 리스트와 같이 처리된다. 문자열의 초기화는 따옴표와 작은 따옴표로 초기화 할 수 있다.
```python
# 초기화
string_data = 'Hello World '

# 문자열 연산
print( string_data + " 수요일 인줄 알았어...")
'''
> Hello World 수요일 인줄 알았어...
'''
print( string_data * 3)
'''
> Hello World Hello World Hello World 
'''
print( string_data[1:6] )
'''
> ello
'''
```


### 튜플
튜플은 리스트와 비슷하지만 초기 선언 이후 내용을 변경할 수 없습니다. 튜플은 초기화 시에 대괄호가 아닌 소괄호를 사용한다.

간단하게 알아본 튜플의 유용한 점은 다음과 같다.
- 메모리 가용성. 리스트에서 언급했듯이 리스트는 연결 리스트의 구조를 갖고 있기 때문에 메모리를 많이 사용하게 된다. 하지만 튜플은 고정 메모리 영역을 갖게 되므로 제한된 메모리만을 사용한다.
- 불변성. 변하지 않는 내용의 정보를 담을 때 사용한다. 그래프를 구현할 때, (Node, Cost) 와 같이 구성할 수 있다.

```python
# 흔하게 사용하는 튜플의 구조.
# 다른 자료형 안에 들어가있다.
data = [(1,2), (2,3), (3,4)]

# 튜플 초기화
tuple_a = (1,2,3,4,5)
print(tuple_a)
'''
> (1, 2, 3, 4, 5)
'''
```

### Dictionary(사전) 자료형
Dictionary는 Key-Value를 쌍으로 가지는 자료형. 리스트의 인덱스와 달리 Key는 순서가 없고 Key를 Hash Table에 저장하기 때문에 자료의 추가와 삭제, 검색 등에 있어 O(1)시간에 처리가 가능하다. 
```Python
# 딕셔너리 초기화
# {Key1:Value1, Key2:Value2, Key3:Value3, ...}
k_bbong = {"김연아":"피겨스케이팅", "류현진":"야구", "박지성":"축구", "BTS":"빌보드"}
print(k_bbong)

# 딕셔너리에 키-밸류 입력

a = {1: 'a'}
a[2] = 'b'
a['bts'] = 'billboard'
a['list'] = [1,2,3]
a['tuple'] = (1,30)
print(a)
'''
> {1: 'a', 2: 'b', 'bts': 'billboard', 'list': [1, 2, 3], 'tuple': (1, 30)}
'''

```


## 조건문
조건문은 '참, 거짓'을 판별하여 지정된 코드를 수행한다. 기본적인 구조는 아래와 같다.
```Python
if (condition1) :
    condition1_code
elif (condition2) :
    condition2_code
else :
    else_code
```
상기해둔 condition1, condition2에는 참 거짓을 판별할 수 있는 조건(Bool Type 을 반환하는 연산)이 들어간다.
때문에 True, False를 반환하는 조건이 들어가야한다. 다른 자료형이 들어갈 경우, 0을 제외하곤 모두 참으로 취급한다.
대표적인 Bool Type 반환하는 연산자 들은 다음과 같다.

### 비교연산자
- X == Y ; XY 같음
- X != Y ; XY 다름
- X > Y ; X가 Y 초과
- X < Y ; X가 Y 미만
- X >= Y ; X가 Y 이상.
- X <= Y ; X가 Y 이하

### 논리연산자
- X and Y ; 모두 참 일때 참.
- X or Y ; 둘 중 하나만 참이어도 참.
- not X ; Bool의 반대를 리턴.

### in / not in
```python
# 문자열에 특정 문자가 있는지 확인
test = "This is test"
if "This" in test:
    print(test)
```

## 반복문
반복문은 while과 for문으로 나뉜다.

### while문
while문은 if문과 비슷하게 조건을 판단하여 해당 조건이 참이면 내부의 코드를 반복하여 실행한다.
```Python
# 기본 문법
while (condition):
    repeat_the_codes

# 반복문의 처음으로 돌아가기
while (condition):
    repeat_the_code1
    if (wanna_go_first):
        continue # 처음으로 돌아감.
    repeat_the_code2

# 반복문 빠져나가기
while (condition):
    repeat_the_code1
    if (wanna_go_out):
        break # 반복문을 빠져 나감.
    repeat_the_code2
```

### for문
for문은 다음과 같은 구조를 갖는다.
```Python
for (var) in (list):
    repeat_the_code
```
변수를 리스트에서 확인한다. 따라서 in 뒤에는 index를 가지는 자료형(list, tuple, string)이 온다.

```Python
# n 회 반복하는 for문
n = 10
for _ in range(n):
    print("count")

# 리스트의 항목들을 출력
list_sample= [1,2,3,4,5,6,7]
for item in list_sample:
	print(item)
'''
> 1
2
3
4
5
6
7
'''
```

---
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
