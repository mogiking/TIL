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

### Timer
매초 음의 리워드로 보상, 도착시 큰 양의 리워드로 보상



## Reward Function

### IS straight?
```python
A = 
B
C
```
### Timer