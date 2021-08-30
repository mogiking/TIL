# Linear Regression
지도 학습의 다른 분류인 선형 회귀는 데이터의 분류 작업이 아닌 연속적인 변수에 대한 출력값을 예측합니다. 

## 단순 선형 회귀
단순 선형 회귀는 하나의 특성(변수)와 연속적인 타겟(y) 사이의 관계를 모델링 한다.
$$ y = w_0 + w_1x $$
$w_0$는 y 절편, $w_1$은 $x$의 가중치 이다.

데이터에 가장 잘 맞는 $f(x)$를 Regression Line(회귀 직선)이라고 하고, 회귀 직선과 샘플 사이의 직선 거리를 Offset(오프셋) 또는 예측 오차인 Residual(잔차) 라고 한다.

## 다중 선형 회귀
회귀의 특성이 하나가 아닌 경우, 이를 다중 선형 회귀라고 한다.
$$ y = w_0x_0 + w_1x_1 + \cdots + w_mx_m $$
예를 들어 2개의 특성을 가지고 있는 모델의 경우 2차원의 평면이 나타납니다. 특성이 2개가 넘어가게 되면 표현이 시각화가 어렵습니다.

## 특성 시각화 및 EDA

EDA(Exploratory Data Analysis, 탐색적 데이터 분석)은 머신러닝 훈련 전 수행하는 중요하고 권장하는 단계dl다. 

### Scatterplot
Scatterplot matrix로 특성 간의 상관관계를 한번에 시각화 할 수 있다.

```Python
# 주택 데이터 load
import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/rasbt/'
                 'python-machine-learning-book-3rd-edition/'
                 'master/ch10/housing.data.txt',
                 header=None,
                 sep='\s+')

df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 
              'NOX', 'RM', 'AGE', 'DIS', 'RAD', 
              'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df.head()
```


```shell
> pip install mlxtend
```


```Python
import matplotlib.pyplot as plt
from mlxtend.plotting import scatterplotmatrix

cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']

scatterplotmatrix(df[cols].values, figsize=(10, 8), 
                  names=cols, alpha=0.5)
plt.tight_layout()
# plt.savefig('images/10_03.png', dpi=300)
plt.show()
```

### 상관관계 행렬 분석(피어슨 상관계수)
상관관계 행렬은 각 변수간의 상관관계를 피어슨 상관계수로 나타낸 정방행렬로, 피어슨의 상관관계는 $-1<=r<=1$까지로 $r=1$이면 두 특성이 완벽한 양의 상관관계, $r=-1$이면 두 특성이 완벽한 음의 상관관계이다. $r=0$일때, 아무런 상관관계가 없다.

```Python
import numpy as np
from mlxtend.plotting import heatmap


cm = np.corrcoef(df[cols].values.T)
hm = heatmap(cm, row_names=cols, column_names=cols)

# plt.savefig('images/10_04.png', dpi=300)
plt.show()
```

## 회귀 모델의 파라미터 구하기
인공 뉴런에서 사용했던 Adaline과 같은 방법으로 Stochastic Gradient Descent(SGD)를 사용하여 최적화를 진행한다. 이때 비용함수는 아달린과 같이 제곱오차합(SSE)이다.

```Python
# 사이킷 런에서 회귀 모델 가중치 추정
from sklearn.linear_model import LinearRegression

slr = LinearRegression()
slr.fit(X, y)
y_pred = slr.predict(X)
print('기울기: %.3f' % slr.coef_[0])
print('절편: %.3f' % slr.intercept_)

lin_regplot(X, y, slr)
plt.xlabel('Average number of rooms [RM]')
plt.ylabel('Price in $1000s [MEDV]')

# plt.savefig('images/10_07.png', dpi=300)
plt.show()
```
연립 1차 방정식으로 문제를 푸는 방법도 있지만, 이 경우 역행렬을 구하는데 너무 많은 비용이 소비가 된다.

## 회귀모델의 성능평가 지표

### Residual plot
회귀 모델에서 잔차를 분포도로 그립니다. 완벽한 모델의 경우 $y=0$ 그래프가 나오겠지만, 거의 모든 경우 0에서 분포하는 형태로 그래프가 그려지게 됩니다.

### 결정계수($R^2$)

**Mean Square Error 평균제곱오차**
MSE는 예측값과 실제값의 차이(오차)의 제곱의 평균.
$$ R^2 = 1 - frac{MSE}{Var(y)}$$
결정계수는 MSE를 표준화한 것으로 해석할 수 있다. 

## 회귀에 규제 적용
가장 널리 사용되는 규제는 **Ridge Regression(릿지회귀), LASSO(Least Absolute Shrinkage and Selection Operator), Elastic Net(엘리스틱 넷)** 이다.

### Ridge Regression
릿지 회귀는 단순히 MSE 함수에 가중치 제곱합을 추가한 L2 규제이다.
$$\lambda||w||^2_2$$

### LASSO
$$\lambda||w||_1$$
- LASSO 는 규제 강도에 따라서 어떤 가중치는 0이 될 수 있습니다.
- LASSO 는 L1 페널티 모델입니다.
- LASSO 는 m>n(n은 훈련 샘플)의 경우 최대 n개의 특성을 선택할 수 있는 한계가 있습니다.

### Elasticnet
$$\lambda_1||w||^2_2 + \lambda_2||w||_1$$
엘라스틱 넷은 둘의 절충안으로 사용합니다. 희소 모델을 만드는 L1과 많은 특성을 선택할 수 있는 L2가 모두 들어있습니다.

## Random Forest를 사용하여 비선형 관계 다루기
결정트리는 Classification에서 주로 사용하던 알고리즘이다. 
결정트리의 불순도 지표를 엔트로피로 정의했는데 이것을 MSE로 교체하면 MSE가 가장 낮게 나오는 분류 모델을 만든다. 해당 모델이 Regression 모델이 될 수 있다.
결정트리 Regression에선 MSE를 종종 노드 내 분산(within-node variance)라고 부르고, 분할 기준을 분산 감소로 이야기 한다.

```Python
from sklearn.tree import DecisionTreeRegressor

X = df[['LSTAT']].values
y = df['MEDV'].values

tree = DecisionTreeRegressor(max_depth=3)
tree.fit(X, y)

sort_idx = X.flatten().argsort()

lin_regplot(X[sort_idx], y[sort_idx], tree)
plt.xlabel('% lower status of the population [LSTAT]')
plt.ylabel('Price in $1000s [MEDV]')
# plt.savefig('images/10_14.png', dpi=300)
plt.show()
```

**랜덤 포레스트**는 결정트리 모델을 앙상블하는 방법이다. 과적합될 수 있지만 특성간의 관계를 잘 나타낸다.