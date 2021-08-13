# AWS DeepRacer
## Reward Function Logics
양의 리워드(+), 음의 리워드(-)

### IS straight?
연속된 waypoint의 좌표를 계산하여 직선이면 
높은 속도에서 양의 리워드로 보상


A = 바로 앞의 waypoint
B = A 다음 Waypoint
C = A의 n회 이후의 waypoint

theta B = vec{ab}의 각도
theta C = vec{ac}의 각도

alpha = 직선 임계치

if theta B <= theta C +_ alpha:
    heading <= theta C +_ alpha:
        if speed >= high:
            reward
### Left corner & Right Corner
바로 앞의 두 웨이포인트가 좌회전{우회전}일때 스티어링(조향)이 왼쪽{오른쪽}이면 +
### Timer
매초 음의 리워드로 보상, 도착시 큰 양의 리워드로 보상



## Reward Function

### IS straight?
```python
a = (x1,y1) #nearest waypoint
b = (x2,y2) #nearest +1 waypoint
c = (x3,y3) #nearest +n waypoint

theta_b = math.atan2(x2-x1,y2-y1)
theta_c = math.atan2(x3-x1,y3-y1)

high = 2

if (theta_c - alpha < theta_b) or (theta_c - alpha > theta_b):
	if (theta_c - beta < heading) or (theta_c - beta > heading):
		if speed > high:
			return += 10
		else:
			return += 1e-3

```
### Timer
