# Ch4. 구현

## Page 115. 왕실의 나이트

```python
input_list = ["A","8"]#Input()
 
ydic={
    "A" : 1, "B" : 2, "C" : 3, "D" : 4,
    "E" : 5, "F" : 6, "G" : 7, "H" : 8,
    1: "A", 2: "B", 3: "C", 4: "D",
    5: "E", 6: "F", 7: "G", 8: "H"
}

x = int(input_list[1])
y = ydic[input_list[0]]
 
print_list = [(x-1,y+2),(x+1,y+2),(x-1,y-2),(x+1,y-2),
              (x+2,y-1),(x+2,y+1),(x-2,y-1),(x-2,y+1)]

for i in print_list:
    if (i[0] < 1 or i[0] > 8):
        continue
    if (i[1] < 1 or i[1] > 8):
        continue
    print(ydic[i[1]],i[0])
```

## Page 118. 게임개발

현민이는 게임 캐릭터가 맵 안에서 움직이는 시스템을 개발중이다. 
캐릭터가 있는 장소는 1x1 크기의 정사각형으로 이뤄진 NxM 크기의 직사각형으로, 각각의 칸은 육지 또는 바다이다. 
캐릭터는 동서남북 중 한 곳을 바라본다.

맵의 각 칸은 (A,B)로 나타낼 수 있고, A는 북쪽으로 부터 떨어진 칸의 개수, B는 서쪽으로부터 떨어진 칸의 개수이다.
캐릭터는 상하좌우로 움직일 수 있고 바다로 되어있는 공간에는 갈 수 없다.
캐릭터의 움직임을 설정하기 위해 정해 놓은 매뉴얼은 이러하다.

1.  현재 위치에서 현재 방향을 기준으로 왼쪽방향 부터 차례대로 갈 곳을 정한다.
2.  캐릭터의 왼쪽 방향에 아직 가보지 않은 칸이 존재한다면, 왼쪽 방향으로 횐전한 다음 왼쪽으로 한칸 전진한다.
    캐릭터의 바로 왼쪽 방향에 아직 가보지 않은 칸이 존재하지 않는다면, 왼쪽 방향으로 회전만 수행하고 1단계로 돌아간다.
3.  만약 네 방향 모두 이미 가본 칸이거나 바다로 되어 있는 칸의 경우에는 바라보는 방향을 유지한 채로 한칸 뒤로가고 1단계로 돌아간다.
    단, 이때 뒤쪽 방향이 바다인 칸이라 뒤로 갈 수 없는 경우에는 움직임을 멈춘다.

현민이는 위 과정을 반복적으로 수행하면서 캐릭터의 움직임에 이상이 있는지를 테스트하려고 한다. 매뉴얼에 따라 캐릭터를 이동시킨 뒤에, 캐릭터가 방문한 칸의 수를 출력하는 프로그램을 만드시오.

입력조건:
- 첫째 줄에 맵의 세로크기와 가로크기를 공백으로 구분하여 입력한다.
- 둘째줄에 게임캐릭터가 있는 칸의 좌표와 바라보는 방향이 각각 서로 공백으로 구분하여 주어진다 방향 d 의 값은 다음과 같이 4가지가 존재한다.
- 셋째줄부터 맵이 육지인지 바다인지에 대한 정보가 주어진다. 0 육지 1 바다.
- 처음에 캐릭터가 위치한 칸의 상태는 항상 육지이다.

```python
# Input 정의
'''
4 4
1 1 0
1 1 1 1
1 0 0 1
1 1 0 1
1 1 1 1
'''

# 맵 크기 설정
n,m = 4,4 # input

# 캐릭터의 x,y 좌표값 변수 초기화.
x,y,view = 1,1,0 # input

array = []
for i in range(n):
    array.append(list(map(int, input().split())))

# 방향 변수 설정
d = {
    0:(0,-1),
    1:(1,0),
    2:(0,1),
    3:(-1,0)
}

# Map의 방문 여부 변수 초기화.
# mapinfo input = (x좌표,y좌표)
# mapinfo output = [방문횟수, 바다 = 1 / 땅 = 0]
mapinfo = {}
for i in range(n):
    for k in range(m):
        mapinfo[(i,k)] = [0, array[i][k]] 

# 캐릭터 회전
def rotate():
    global view
    view = ((view-1)%4)

def go():
    global x,y,d,view,mapinfo
    x += d[view][0]
    y += d[view][1]
    mapinfo[x,y][0] = 1

def search():
    global x,y,d,view,mapinfo
    #Map을 벗어나는지 확인
    if (x+d[(view-1)%4][0])  < 0  or (x+d[(view-1)%4][0]) > n :
        return False
    if (y+d[(view-1)%4][1]) < 0 or (y+d[(view-1)%4][1]) > m:
        return False
    #방문 여부 확인
    if mapinfo[(x+d[(view-1)%4][0],y+d[(view-1)%4][1])][0] >= 1:
        return False
    #바다인지(갈 수 있는지 확인)
    if mapinfo[(x+d[(view-1)%4][0],y+d[(view-1)%4][1])][1] == 1:
        return False
    return True

def go_back():
    global x,y,d,view,mapinfo
    # 바다인지 확인
    if mapinfo[(x+d[(view-2)%4][0],y+d[(view-2)%4][1])][1] == 1:
        return False
    else : 
        x += d[(view-2)%4][0]
        y += d[(view-2)%4][1]
        return True

def count_visit():
    global x,y,d,view,mapinfo,n,m
    _count = 0

    for i in range(n):
        for k in range(m):
            if mapinfo[(i,k)][0] == 1:
                _count += 1

    return _count

def game():
    _rotate_count = 0
    while True:
        if search():
            rotate()
            go()
            _rotate_count = 0
        else :
            rotate()
            _rotate_count += 1
        if _rotate_count == 4:
            _rotate_count = 0
            if not go_back():
                return count_visit()

print(game())

```