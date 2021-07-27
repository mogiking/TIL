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
