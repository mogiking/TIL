''' p.149 음료수 얼려먹기
NxM 크기의 얼음 틀이 있다. 구멍이 뚫려있는 부분은 0, 칸막이가 존재하는 부분은 1로 표시된다.
구멍이 뚫려 있는 부분끼리  상,하,좌,우로 붙어 있는 경우 서로 연결되어 있는 것으로 간주한다. 
이때 얼음틀의 모양이 주어졌을 때, 생성되는 총 아이스크림의 개수를 구하는 프로그램을 작성하시오.
'''
import random

n, m = 3,4 #input()
map_info = dict()
random.randint(0,1)

for i in range(n):
    for j in range(m):
        map_info[(i,j)] = [random.randint(0,1),0]

def print_map():
    for i in range(n):
        for j in range(m):
            print(map_info[(i,j)][0],end = ' ')
        print()

def search(i,j):
    global n,m
    if (map_info[(i,j)][0] == 0) and (map_info[(i,j)][1] == 0):
        map_info[(i,j)][1] = 1
        if (i < n-1):
            search(i+1,j)
        if (j < m-1):
            search(i,j+1)
        if (i > 0):
            search(i-1,j)
        if (j > 0):
            search(i,j-1)
    return

def search_all():
    count = 0
    global n,m
    for i in range(n):
        for j in range(m):
            if (map_info[(i,j)][0] == 0) and (map_info[(i,j)][1] == 0):
                search(i,j)
                count += 1
    return count

print_map()
print (search_all())

''' p.152 미로 탈출
동빈이는 NxM 크기의 직사각형 형태의 미로에 갇혀 있다. 
미로에는 여러 마리의 괴물이 있어 이를 피해 탈출해야 한다. 
동빈이의 위치는 (1,1)이고 미로의 출구는 (N,M)의 위치에 존재하며 한번에 한 칸씩 이동할 수 있다.
이때 괴물이 있는 부분은 0 으로 괴물이 없는 부분은 1로 표시되어 있다.
미로는 반드시 탈출할 수 있는 형태로 제시된다. 이때 동빈이가 탈출하기 위해 움직여야 하는 최소 칸의 개수를 구하시오.
칸을 셀 때는 시작 칸과 마지막 칸을 모두 포함해서 계산한다.
'''

n, m = 3,4 #input()
map_info = dict()
