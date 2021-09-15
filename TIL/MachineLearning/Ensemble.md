# Ensemble (앙상블)

- 다수결 투표를 통한 예측 만들기
- 중복을 허용하는 Bagging을 사용하여 과대적합 감소
- 약한 학습기(weak learner)로 구성된 부스팅(Boosting)으로  모델 구축하기

# 앙상블 학습

 앙상블 학습은 개별 분류기를 하나의 메타 분류기로 연결하여 개별 분류기보다 더 좋은 일반화 성능을 달성하는 것 입니다. 다수결 투표(plurality voting)를 통해 가장 많은 선택을 받은 클래스 레이블을 선택합니다. 유명한 앙상블 방법 중 하나로는 서로 다른 결정 트리(Decision Tree)를 연결한 랜덤 포레스트(Random Forest)가 있습니다.

 각 모델이 독립적이라는 가정하에 에러율($\varepsilon$) 다음과 같습니다.

$$P( y\ge k ) = \sum_{k}^{n} \binom{n}{k} \varepsilon^k(1-\varepsilon)^{n-k}=\varepsilon_{ensemble}$$

 에러율($\varepsilon$)이 0.25인 분류기 11개로 구성된 앙상블의 에러율은 0.034로 현격히 낮아집니다. 개별 에러율이 0.5 이하일 경우, 앙상블의 에러율이 일반 에러율보다 낮습니다.

# 배깅(Bagging)

 원본 훈련 데이터셋에서 부트스트랩(Bootstrap) 샘플(중복을 허용한 랜덤 샘플)을 뽑아서 사용합니다. 따라서 배깅을 bootstrap aggregating이라고 합니다. 고차원 데이터셋을 사용하는 더 복잡한 문제라면 단일 결정트리가 쉽게 과대적합 됩니다. 배깅 알고리즘은 모델의 분산을 감소하는 효과적인 방법이지만, 모델의 편향을 낮추는 데는 효과적이지 않습니다. 

따라서 배깅을 수행할 때, 편향이 낮은 모델, 예를 들어 가지치기하지 않은 결정 트리를 분류기로 하여 앙상블을 만드는 이유입니다.

# 부스팅(Boosting)

 부스팅에서 앙상블은 약한 학습기(weak learner)라고도 하는 간단한 분류기로 구성합니다. 예를 들어 약한 학습기는 깊이가 1인 결정트리 입니다.

1. 훈련 데이터셋 $D$에서 중복을 허용하지 않고 랜덤한 부분 집합 $d_1$을 봅아 약한 학습기 $C_1$을 훈련합니다.
2. 훈련 데이터셋에서 중복을 허용하지 않고 두 번째 랜덤한 부분 집합 $d_2$를 뽑고 이전에 잘못 분류된 샘플의 50%를 더해서 약한 학습기 $C_2$를 훈련합니다.
3. 훈련 데이터셋 에서 $C_1$과 $C_2$에서 잘못 분류한 훈련 샘플 $d_3$를 찾아 세번째 약한 학습기인 $C_3$를 훈련합니다
4. 약한 학습기 $C_1,C_2,C_3$를 다수결 투표로 연결합니다.

부스팅은 배깅 모델에 비해 분산은 물론 편향도 감소시킬 수 있습니다. 실제로는 에이다 부스트 같은 부스팅 알고리즘이 분산이 높다(과대적합)고 알려져 있습니다.