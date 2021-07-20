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

# ### column name to array
# Column 명을 array로 저장합니다.

columns = list(df.columns.values)
columns

# ## 요약 통계 확인
# Data에 대한 각종 통계를 확인합니다.

df.info()

df.describe()

# ### NaN 개수, Null 개수 확인
# 각 칼럼의 NaN, Null 개수를 확인합니다.

df.isna().sum()


