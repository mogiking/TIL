# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Preprocessing 전처리
# 본 페이지에선 사용했던 전처리 프로세스를 기록합니다.

# ### Read + Merge CSV
# 1. 지정된 디렉토리에서 csv들을 가져와 읽습니다.
# 1. csv를 하나의 Dataframe으로 합칩니다.

# +
import pandas as pd

# 경로 및 파일명 지정
normal = "./opcua_data.csv"
abnormal = "./opcua_data_ab.csv"

# DataFrame에 csv를 읽어온 뒤, Index를 재정렬
df = pd.concat(map(pd.read_csv, [normal, abnormal]), ignore_index=True)
df
# -

# ## Column
#
# ### column 정보 확인
# 가져온 Dataframe의 Column 정보를 확인합니다.

df.columns

# ### column 이름 변환
# 가져온 Dataframe의 Column 이름을 바꿉니다.

df = df.rename(columns={'시간': 'time', 
                        '디바이스 이름' : 'device',
                        'measurement_ns5sSinusoid1 => sinusoid1': 'sin1',
                        'measurement_ns5sSawtooth1 => sawtooth1' : 'saw1',
                        'measurement_ns5sTriangle1 => triangle1' : 'tri1'})
df.columns

# ### Column 데이터 형 변환
# object 타입에서 각 데이터 형에 맞게 변환

# +
## column data type 변환
df['time'] = pd.to_datetime(df['time'])

#sin1,saw1,tri1에 각각 strip을 사용하여 ' 를 제거 후 float형으로 변환
df['sin1'] = df['sin1'].str.strip('\'').astype('float')
df['saw1'] = df['saw1'].str.strip('\'').astype('float')
df['tri1'] = df['tri1'].str.strip('\'').astype('float')

df.dtypes
# -

# ### Column의 Value count
# 각 컬럼의 Value의 빈도 확인.

df['sin1'].value_counts()

# ### column 삭제 (drop)

df = df.drop(columns=['device'])

# ### Column 추가 (target Column 추가)

df['target'] = 0
df

# ### column name to array
# Column 명을 array로 저장합니다.

columns = list(df.columns.values)
columns

# ## Masking
# 특정 조건으로 Masking Array를 만든 뒤, 마스킹을 DF에 적용합니다.

# +
#비정상 데이터 구간
#2021.07.05 13:37 ~ 14:18

#마스킹 생성.
#시작시간 마스크와 종료시간 마스크가 같은 부분을 마스킹 함.
start_time = '2021-07-05 04:37:00.000'
end_time = '2021-07-05 05:18:00.000'
mask1 = (df.time > start_time) 
mask2 = (df.time < end_time)
mask = mask1==mask2

#마스킹에 맞게 target 변환
column_name= 'target'
df.loc[mask, column_name] = 1

#입력 확인.
df['target'].value_counts()

# -

# ## 요약 통계 확인
# Data에 대한 각종 통계를 확인합니다.

df.info()

df.describe()

# ### NaN 개수, Null 개수 확인
# 각 칼럼의 NaN, Null 개수를 확인합니다.

df.isna().sum()

# ## NaN 처리
# NaN을 처리하기 위한 샘플 df2 생성

# +
import pandas as pd
from io import StringIO
import sys

csv_data = \
'''A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
10.0,11.0,12.0,'''

# 파이썬 2.7을 사용하는 경우
# 다음과 같이 문자열을 유니코드로 변환해야 합니다:

if (sys.version_info < (3, 0)):
    csv_data = unicode(csv_data)

df2 = pd.read_csv(StringIO(csv_data))
df2
# -

# ### NaN row 삭제. dropna()
# NaN이 존재하는 Row 제거
# - dropna(axis=0) : 0 = NaN이 존재하는 Row 제거. / 1 = NaN이 존재하는 Column 제거.
# - (how='all') : 모든 열이 NaN인 Row 제거.
# - (thresh = 4) : NaN이 아닌 값이 4개보다 적은 Row를 제거.
# - (subset=['A']) : A Column에서 NaN이 존재하면 해당 Row를 제거.

df2.dropna(axis=0)

df2.dropna(axis=1)

df2.dropna(how='all')

df2.dropna(thresh=4)

df2.dropna(subset=['C'])

# ### Interpolation
# 보간. NaN을 다른 값으로 대체
# **sklearn.impute의 SimpleImputer**로 수행.
#
# SimpleImputer(missing_values=np.nan, **strategy='mean'**) #strategy에 들어갈 수 있는 변수
# - mean: 평균
# - median: 중앙값
# - most_frequent: 가장 많이 나타난 값. (범주형 데이터를 다룰때 자주 쓰임)
# - constant: 특정 값으로 채움. fill_value: string or numeric 추가 필요.

# +
# 행의 평균으로 누락된 값 대체하기

from sklearn.impute import SimpleImputer
import numpy as np

imr = SimpleImputer(missing_values=np.nan, strategy='mean')
imr = imr.fit(df2.values)
imputed_data = imr.transform(df2.values)
imputed_data

# + [markdown] id="VLIH95sZMq2Z"
# `SimpleImputer`는 한 특성의 통곗값을 사용하여 누락된 값을 채웁니다. 이와 달리 `IterativeImputer` 클래스는 다른 특성을 사용하여 누락된 값을 예측합니다. 먼저 `initial_strategy` 매개변수에 지정된 방식으로 누락된 값을 초기화합니다. 그다음 누락된 값이 있는 한 특성을 타깃으로 삼고 다른 특성을 사용해 모델을 훈련하여 예측합니다. 이런 식으로 누락된 값이 있는 모든 특성을 순회합니다.
#
# `initial_strategy` 매개변수에 지정할 수 있는 값은 `SimpleImputer`와 동일하게 `'mean'`, `'median'`, `'most_frequent'`, `'constant'`가 가능합니다.
#
# 예측할 특성을 선택하는 순서는 누락된 값이 가장 적은 특성부터 선택하는 `'ascending'`, 누락된 값이 가장 큰 특성부터 선택하는 `'descending'`, 왼쪽에서 오른쪽으로 선택하는 `'roman'`, 오른쪽에서 왼쪽으로 선택하는 `'arabic'`, 랜덤하게 고르는 `'random'`이 있습니다. 기본값은 `'ascending'`입니다.
#
# 특성 예측은 종료 조건을 만족할 때까지 반복합니다. 각 반복 단계에서 이전 단계와 절댓값 차이 중 가장 큰 값이 누락된 값을 제외하고 가장 큰 절댓값에 `tol` 매개변수를 곱한 것보다 작을 경우 종료합니다. `tol` 매개변수 기본값은 1e-3입니다. 또는 `max_iter` 매개변수에서 지정한 횟수에 도달할 때 종료합니다. `max_iter`의 기본값은 10입니다.
#
# 예측에 사용하는 모델은 `estimator` 매개변수에서 지정할 수 있으며 기본적으로 `BayesianRidge` 클래스를 사용합니다. 예측에 사용할 특성 개수는 `n_nearest_features`에서 지정할 수 있으며 상관 계수가 높은 특성을 우선하여 랜덤하게 선택합니다. 기본값은 `None`으로 모든 특성을 사용합니다.
#
# `KNNImputer` 클래스는 K-최근접 이웃 방법을 사용해 누락된 값을 채웁니다. 최근접 이웃의 개수는 `n_neighbors` 매개변수로 지정하며 기본값은 5입니다. 샘플 개수가 `n_neighbors` 보다 작으면 `SimpleImputer(strategy='mean')`과 결과가 같습니다.
#
# **머신러닝 교과서 with 파이썬,사이킷런,텐서플로 개정3판, 박해선 옮김** 발췌

# + colab={"base_uri": "https://localhost:8080/"} id="iuTQp_czMq2Z" outputId="01935bee-7f43-4ac9-c5b9-0216684a73d7"
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

iimr = IterativeImputer()
iimr.fit_transform(df2.values)

# +
from sklearn.impute import KNNImputer

kimr = KNNImputer()
#kimr.fit_transform(df.drop(columns=['time']).values)
# -

# ### 이전 값, 이후 값으로 채우기 fillna()
#
# df**.fillna(method='bfill')**
# - bfill: 다음 행(Row)의 값으로 NaN을 채움.
# - ffill: 이전 행(Row)의 값으로 NaN을 채움.

df.fillna(method='bfill') # method='backfill'와 같습니다

df.fillna(method='ffill') # method='pad'와 같습니다

# # 범주형 데이터 다루기
# **범주형 데이터 : ** 범주형 데이터는 크게 순서가 없는 특성(색), 순서가 있는 특성(사이즈), 수치형 특성(가격)이 있습니다.
# 각각의 특성을 갖는 dataframe 을 생성하겠습니다.

import pandas as pd
df = pd.DataFrame([
    ['green', 'M', 10.1, 'class1'],
    ['red', 'L', 13.5, 'class2'],
    ['blue', 'XL', 15.3, 'class1']
])
df.columns=['color','size','price','classlabel']
df

# ## 순서가 있는 특성 매핑
# 순서가 있는 특성은 범주형의 문자를 정수로 바꿔야 한다. 자동으로 특성에 맞춰 변환 해주는 함수는 없기 때문에 매핑함수를 만들어야 한다.
# $$XL = L + 1 = M + 2$$

size_mapping = {
    'XL':3,
    'L':2,
    'M':1
}
df['size'] = df['size'].map(size_mapping)
df

# 원래의 범주형 데이터로 변환시에는 아래와 같은 함수를 사용하면 된다.

inv_size_mapping = {v: k for k, v in size_mapping.items()}
df['size'].map(inv_size_mapping)

# ## 클래스 레이블 인코딩
# 대부분의 라이브러리 들은 클래스 레이블이 정수로 매핑되어 있다고 가정합니다. sklearn의 경우 대부분 정수로 변환을 해주지만 실수를 줄이기 위해 정수 배열로 전달하는 것이 좋습니다. 클래스 레이블은 순서가 없다는 것이 중요한 특성입니다. enumerate를 사용하여 클래스 레이블을 0부터 할당합니다.

import numpy as np
class_mapping = {label:idx for idx, label in enumerate(np.unique(df['classlabel']))}
class_mapping

df['classlabel'] = df['classlabel'].map(class_mapping)
df

# sklearn의 LabelEncoder를 사용하면 조금 더 편하게 매핑할 수 있습니다.

from sklearn.preprocessing import LabelEncoder
class_le = LabelEncoder()
y = class_le.fit_transform(df['classlabel'].values)
y

# inverse_transform 을 통해 원래 Label로 되돌릴 수 있습니다.

class_le.inverse_transform(y)

# ## 순서가 없는 범주형 데이터
# color의 경우 순서가 없기 때문에 1,2,3 과 같은 정수형 숫자로 변환시에 학습 알고리즘이 특정 색상은 다른 색보다 작거나 크다고 판단할 수 있습니다. 이를 방지하기 위해서 OneHotEncoder를 사용합니다.

from sklearn.preprocessing import OneHotEncoder
X = df[['color','size','price']].values
color_ohe = OneHotEncoder()
color_ohe.fit_transform(X[:,0].reshape(-1,1)).toarray()

# 여러개의 특성이 있는 배열에서 특정 열만 변환하려면 ColumnTransformer를 사용합니다. 이 클래스는 다음과 같이 (name, transformer, columns) 튜플리스트를 받습니다.

from sklearn.compose import ColumnTransformer
X = df[['color','size','price']].values
c_transf = ColumnTransformer([
    ('onehot', OneHotEncoder(), [0]),
    ('nothing', 'passthrough', [1,2])
])
c_transf.fit_transform(X)

# 더 편리한 방법은 Pandas의 get_dummies 메서드를 활용하는 방법입니다.

pd.get_dummies(df[['color','size','price']])

#특정 컬럼을 지정할 수도 있습니다.
pd.get_dummies(df[['color','size','price']],columns=['size'])

# OneHot 인코딩을 사용할 땐 다중공선성(Multicollinearity)를 고려해야합니다. 변수 간의 상관 관계 감소를 위해서 하나의 특성 컬럼을 지우는 방식으로 상관관계를 낮출 수 있습니다. color_blue를 지워도 red,green 모두 0 이라면 blue로 판별.
#
# * get_dummies를 에서 drop_first 매개변수를 True로 지정하면 첫번째 열을 삭제할 수 있습니다.
# * OneHotEncoder에서는 drop='first', categories='auto'로 지정합니다. drop의 매개변수를 'if_binary'로 설정하면 두 개의 범주를 가진 특성일 때만 첫번째 열이 삭제됩니다.

pd.get_dummies(df[['color','size','price']], drop_first=True)

from sklearn.preprocessing import OneHotEncoder
X = df[['color','size','price']].values
color_ohe = OneHotEncoder(categories='auto',drop='first')
color_ohe.fit_transform(X[:,0].reshape(-1,1)).toarray()

# # 훈련/테스트 데이터셋 나누기
# sklearn의 train_test_split 함수를 이용하여 나눕니다.
#
# 실전에서 가장 많이 사용하는 비율은 데이터셋의 크기에 따라 6:4,7:3,8:2 입니다. 대용량 데이터셋의 경우에는 90:10 또는 99:1의 비율로 나누는 것도 적절합니다. 10만개 이상의 훈련 샘플이 있다면 1만개의 테스트 셋만 준비해도 괜찮습니다.
#
# 훈련과 테스트가 완료된 후에는 추가 테스트 셋도 훈련 셋도 포함하여 모델의 성능을 향상시키는 방법이 일반적으로 사용됩니다.

'''
from sklearn.model_selection import train_test_split
X = df.iloc[:, 1:5].values
y = df.iloc[: , -1:].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
'''

# # 특성 스케일 맞추기
#
# Decision Tree, Random forest 는 특성 스케일에 영향을 받지 않는 알고리즘 입니다. 하지만 다른 알고리즘들은 특성의 스케일에 크게 영향을 받기 때문에 특성을 학습이 쉽게 맞출 필요가 있습니다. 크게 **정규화**와 **표준화**를 사용합니다.
#
# 최소-최대 스케일변환은 일반적인 정규화와 다른 특별한 케이스입니다.
# $$x^{(i)}_{norm} = \frac{x^{(i)}-x_{min}}{x_{max}-x_{min}}$$
#

'''MinMaxScaler Sample
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
X_train_norm = mms.fit_transform(X_train)
X_test_norm = mms.transform(X_test)
'''

# 최댓값에서 최소값을 빼서 정규화를 하기 때문에 이상치에 민감하게 반응하게 됩니다.
# 표준화는 분포를 정규 분포와 같게 만듭니다. 최소-최대 스케일 변환에 비해 이상치에 덜 민감하게 반응하게 됩니다.
# $$x^{(i)}_{std} = \frac{x^{(i)}-\mu_x}{\sigma_x}$$

'''Standard Scaler Sample
from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)
'''

# 이상치가 많이 포함된 작은 데이터셋을 다룰때는 RobustScaler를 사용한다. RobustScaler는 중간 값($q_2$)를 빼고 1사분위수($q_1$)와 3사분위수($q_3$)의 차이로 나누어 스케일을 조정합니다.
# $$x^{(i)}_{robust} = \frac{x^{(i)}-q_2}{q_3-q_1}$$

'''RobustScaler Sample
from sklearn.preprocessing import RobustScaler
rbs = RobustScaler
X_train_rbs = rbs.fit_transform(X_train)
X_test_rbs = rbs.transform(X_test)
'''

# ## 유용한 특성 선택
# L1,L2 Regularization: overfitting이 되면 상관관계가 크게 늘어남. 이를 방지하기 위함. 
# $$E(w) = \frac{1}{2}\sum_{n = 1}^{N}\{t_n-w^T\phi(x_n)\}^2 + \lambda \frac{1}{2}w^Tw  $$
# $(t_n-w^T\phi(x_n))$에서 $w$가 커지면 뒤의 $(\frac{1}{2}w^Tw)$로 인해 Cost 함수가 커지기 때문에 과적합을 방지할 수 있다.
#
# ### L2 Regularization
# 규제식에 $\lambda$를 곱하는 형식이기 때문에 $\lambda$가 커지면 규제식의 범위도 넓어지기 때문에 과적합이 일어날 수 있다. 아래의 그림에서 면적이 넓진다고 이해하면 되겠다.

from IPython.display import Image
Image(url='https://git.io/JtY8L', width=500) 

# ### L1 Regularization
# L1 규제의 경우 마름모 모양이기 때문에 특정 w가 사라지는 희소성이 나타나게 된다. 아래의 그림을 보면 이해가 쉽다.

Image(url='https://git.io/JtY8t', width=500) 

# ### 순차특성 선택 알고리즘(차원축소)
# 모델 복잡도를 줄이고 과대적합을 피하는 다른 방법은 특성 선택을 통한 차원축소 입니다. 규제가 없는 모델에서 특히 유용합니다.
# - 특성 선택 : 원본 특성에서 일부 선택
# - 특성 추출 : 특성에서 얻은 정보로 새로운 정보를 생성
#
# **순차특성선택** 알고리즘은 Greedy한 방법으로 초기 d차원의 공간을 k 차원으로 축소합니다. 전통적인 순차특성선택 알고리즘은 순차후진선택(Sequential Backward Selection, SBS)입니다. 
#
# 1. 알고리즘을 $k=d$로 초기화합니다. $d$는 전체 특성 공간 $X_d$의 차원입니다.
# 1. 조건 $x^- = argmax J(X_k-x)$를 최대화하는 특성 $x^-$를 결정합니다. 
# 1. 특성 집합에서 특성 $x^-$를 제거합니다.
# 1. k가 목표하는 특성 개수가 되면 종료.
#
# 데이터 수집 비용이 높은 실전 애플리케이션에서는 특성을 줄이는 것이 유용할 수 있습니다. 또 특성 개수를 크게 줄였기 때문에 더 간단한 모델을 얻었고 해석하기도 쉽습니다.

# ### 랜덤 포레스트의 특성 중요도 사용
# 랜덤포레스트를 사용하면 앙상블에 참여한 모든 결정트리에서 계산한 평균적인 불순도 감소로 특성 중요도를 측정할 수 있습니다.RFE는 재귀적 특성 제거 방법을 사용합니다. 처음에 모든 특성을 사용하여 모델을 만들고 특성 중요도가 가장 낮은 특성을 제거합니다. 그다음 제외된 특성을 빼고 나머지 특성으로 새로운 모델을 만듭니다. 이런식으로 미리 정의한 특성 개수가 남을때까지 반복합니다. 
# - n_features_to_select = $n$으로 매개변수에 특성의 갯수를 지정할 수 있습니다.
# - step 매개변수에서 $[0,1]$의 범위에서 실수를 지정하여 비율을 지정할 수 있습니다. 기본 값은 1 입니다.

# # 차원축소
#
#
# ## 주성분 분석을 통한 비지도 차원 축소
# 지나치게 많은 데이터로 인해서 연산속도와 정확도가 떨어지는 차원의 저주(curse of dimensionality)에 빠지게 됩니다. 중요한 원소들만 추출하여 성능을 향상 시킵니다.
#
# ### 주성분 분석의 주요 단계
# PCA는 고차원 데이터에서 분산이 가장 큰 방향(직교), 좀 더 작거가 같은 수의 차원을 갖는 새로운 부분 공간으로 투영한다.
# 새로운 부분공간의 직교좌표는 분산이 최대인 방향으로 해석할 수 있다.
#
# - 차원 축소 전 $x_1,x_2$
# - 차원 축소 후 $PC1,PC2$

Image(url='https://git.io/JtsvW', width=400) 

# PCA는 차원 축소를 위해 $d * k$ 차원의 변환 행렬 $W$를 만듭니다.
# 특성 벡터 $x$를 $W$를 통해 $k$차원의 특성 부분 공간으로 매핑합니다.
# $$W \in \mathbb{R^{d*k}} \\ xW = z \\ z = [z_1,z_2,\cdots,z_k],\ \ z \in \mathbb{R^k}$$
#
# **PCA는 특성 스케일에 민감하게 반응하기 때문에, PCA 특성을 표준화 전처리 해야합니다.**
#
# 1. $d$차원 데이터셋을 표준화 전처리합니다.
# 1. 공분산 행렬(covariance matrix)을 만듭니다.
# 1. 공분산 행렬을 고유 벡터(eigenvector)와 고윳값(eigenvalue)으로 분해합니다.
# 1. 고윳값을 내림차순으로 정렬하고 그에 해당하는 고유 벡터의 순위를 매깁니다.
# 1. 고윳값이 가장 큰 k개의 고유 벡터를 선택합니다. ($k \leq d$)
# 1. 최상위 k개의 고유 벡터로 투영 행렬(projection Matrix) $W$를 만듭니다.
# 1. 투영행렬 $W$를 사용해서 $d$차원 입력 데이터셋 $X$를 새로운 $k$ 차원의 특성 부분 공간으로 변환합니다.

# +
import pandas as pd

df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
# -

from sklearn.model_selection import train_test_split
X, y = df_wine.iloc[:,1:].values, df_wine.iloc[:,0].values
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, stratify=y, random_state=0)

# 1. 특성을 표준화 전처리
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

# 공분산 행렬의 계산은 다음과 같이 계산 합니다.
# $d*d$차원의 대칭행렬로 만듭니다. 공분산은 다음과 같이 계산합니다.
#
# $$ \sigma_{jk} = \frac{1}{n-1}\Sigma_{i=1}^{n}(x_j^{(i)}-\mu _j)(x_k^{(i)}-\mu _k)$$
# 공분산 행렬 $\Sigma$ 에서 고유벡터($v$)와 고유값($\lambda$)을 추출합니다.
# $$ \Sigma v = \lambda v $$

# 2. 공분산 행렬(covariance matrix)을 만듭니다.
import numpy as np
cov_mat = np.cov(X_train_std.T)
print(cov_mat.shape)
# 3. 공분산 행렬을 고유 벡터(eigenvector)와 고윳값(eigenvalue)으로 분해합니다.
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
# 4. 고윳값을 내림차순으로 정렬하고 그에 해당하는 고유 벡터의 순위를 매깁니다.
print('\n 고윳값 \n%s' % eigen_vals)

# 데이터 셋의 특성 부분 공간으로 압축하기 위해서 가장 많은 정보를 가진 고유벡터 일부만 선택합니다. 설명된 분산 비율은  전체 고윳값의 합에서 고윳값의 비율입니다.
# $$\text{설명된 분산 비율 = } \frac{\lambda _j}{\Sigma _{j=1}^{d} \lambda _j}$$

tot = sum(eigen_vals)
var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)


# +
import matplotlib.pyplot as plt

# 5. 고윳값이 가장 큰 k개의 고유 벡터를 선택합니다. ( k≤dk≤d )
plt.bar(range(1, 14), var_exp, alpha=0.5, align='center',
        label='Individual explained variance')
plt.step(range(1, 14), cum_var_exp, where='mid',
         label='Cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.tight_layout()
# plt.savefig('images/05_02.png', dpi=300)
plt.show()
# -

# 부분 공간으로 치환을 하기위해 두개의 특성을 고릅니다.
#
# 고윳값의 내림차순으로 고유 벡터와 고윳값의 쌍을 정렬 후,
# 투영행렬 $W$를 만듭니다.

# +
# (고윳값, 고유벡터) 튜플의 리스트를 만듭니다
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i])
               for i in range(len(eigen_vals))]

# 높은 값에서 낮은 값으로 (고윳값, 고유벡터) 튜플을 정렬합니다
eigen_pairs.sort(key=lambda k: k[0], reverse=True)
# -

# 6. 투영행렬 W 를 만듭니다.
w = np.hstack((eigen_pairs[0][1][:, np.newaxis],
               eigen_pairs[1][1][:, np.newaxis]))
print('투영 행렬 W:\n', w)

# **예제에서는 2가지의 특성만 추출했지만 실제 환경에선 성능에 맞춰 고윳값을 선택해야합니다.**
#
# 전체 데이터셋 $X$에 투영행렬 $W$를 투영(점곱)합니다.
# $$ X' = XW $$

X_train_pca = X_train_std.dot(w)

# +
# 시각화
X_train_pca = X_train_std.dot(w)
colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']

for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_pca[y_train == l, 0], 
                X_train_pca[y_train == l, 1], 
                c=c, label=l, marker=m)

plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
# plt.savefig('images/05_03.png', dpi=300)
plt.show()
