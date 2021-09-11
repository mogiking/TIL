##################  p.92 큰 수의 법칙.##################
# 입력 N, M, K 
# N(2~1000), M(1~10000), K(1~10000)
# K <= M
'''
n, m, k = map(int, input().split())
'''
n, m, k = 3, 5, 2

# N개 의 자연수 입력
# 각 숫자는 1~10000
'''
nums = list(map(int, input().split()))
'''
nums = [3,6,7]

# 입력 받은 자연수 정렬
nums.sort(reverse=True)

# 가장 큰 수(first), 두번째로 큰 수(second) 지정
first = nums[0]
second = nums[1]
result = 0

# 두번째로 큰 수가 더해지는 횟수 계산
# result += second * M/(K+1)
result += second * (m//(k+1))

# 가장 큰 수가 더해지는 횟수 계산
# result += first  * ((M/(K+1)*K) + (M%(K+1)))
result += first * ((m//(k+1)) * k + m%(k+1))

# 7 7 6 7 7 = 34
print(result)


#######################################################


#################  p.96 숫자 카드 게임.#################

# N x M 행렬 사이즈 입력
# N,M (1~100)

# M개의 1 ~ 10000 사이의 자연수 입력. N 회
# N x M 행렬 = card_set

arr = [[1,2,3,4],[4,5,25,6],[7,64,23,1]]
# N개의 배열 초기화
'''
arr = [list(map(int, input().split())) for _ in range(n)]
'''
# list minarr
minarr = list()

# 각 행의 최소값 minarr에 입력
for col in arr:
    minarr.append(min(col))
# min 배열에서 가장 큰 값 출력
result = max(minarr)
# max(min)
print(result)

arr = [[1,2,3,4],[4,5,25,6],[7,64,23,1]]
minarr = list()
for col in arr:
    minarr.append(min(col))
result = max(minarr)
print(result)

#######################################################

#################  p.99 1이 될 때 까지 #################

'''
가능한 행동
1. N 에서 1 빼기
2. N 에서 K 나누기
'''

# N, K 입력

# while N != 1:
#  while N%K != 0:
#   N -= 1
#   count += 1
#  N = N/K
#  count += 1

'''
while True:
    count + = N%K
    if N<K:
        break
    count += 1
    N = int(N/K)
count += N-1
'''
n = 100000
k = 3

count = 0
goal = n
while goal >= k :
    count += (goal%k)
    goal //= k
    count += 1
count += (goal-1)
print(count)

#######################################################
