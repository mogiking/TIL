# TextMining
'21.2학기 텍스트마이닝 수업의 강의 노트 입니다.

# 1강.
전체적인 텍스트 마이닝 정리.
- Data Mining
- Text Mining
- Need for text mining
- Text mining Techniques
- Text mining process
- General architecture of text mning systems
- Text mining applications

## Data Mining
데이터에서 지식을 발견하는 것.
데이터 마이닝은 대용량의 데이터베이스로부터 정보(의미있는, 유용한, 이전에 알려지지 않은, 이해하기 쉬운)를 자동으로 추출하는 과정으로 이해된다.

**Descriptive**
- 데이터로 부터 패턴이나 트렌드를 찾는것.
- 구현: 클러스터링

**Predictive**
- 측정하지 않은, 보지 않은 값을 예측하는 것
- 구현: 군집화

### 표준 프로세스
1. Business Understanding 비즈니스의 이해
1. Data Understanding 데이터 이해
1. Data Preparation 데이터 준비
1. Modeling 모델링
1. Evaluation 평가
1. Deployment 실행

## Text Mining
Textual documnet에서 정보를 발견하는 과정. 흥미로운 패턴과 관계를 자동으로 인식하는 과정.
구조화되지 않은 문서로 데이터를 추출하기 때문에 데이터마이닝 보다 어렵다.

- 통계적 NLP + Data Mining

Data Retrieval : Search(찾고자 하는 것이 명확함) + Structured Data ; SQL 쿼리 
Data Mining: Discover(우연히 발견) + UnStructured Data
Information Retrieval : Search(찾고자 하는 것이 명확함) + Structured ; Data구글 검색
Text Mining: Discover(우연히 발견) + UnStructured Data

**Process**
1. Text
2. Text Preprocessing
    - 자연어처리의 결과물.
    - 문장 경계 인식.
    - Word Tokenization
    - Part-of-Speech tagging (품사를 결정)
    - +Word sense disambiguation (단어의 이미를 결정하는 것)
    - +Parsing (문장 구조를 만듦. Parse Tree)
3. Text Transformation(Feature generation)
    - Text 문서를 단어를 이용해서 표현함. 문서에서 단어가 몇번 나왔는지 feature로 표현함.
    - 단어의 순서는 사용하기 쉽지 않음. (bag-of-words; 단어들을 주머니에 넣음. 배열 순서가 무너짐.)
    - Stopword removal (관사 전치사, function word(common word)를 제거 / HTML 등의 태그를 제거)
4. Feature Selelction
    - 모델을 만들 때 중요한 feature만 선택함.
    - 학습 시간을 줄일 수 있음.
    - 모델의 성능 향상.
    - Feature의 차원을 줄일 수 있음.
5. Data Mining / Pattern Discovery
    - Classification(Supervised Learning)
        - Given : 학습 데이터를 줌. (Labeled records)
        - Find : a Model.
        - Goal : 새로운 데이터가 들어왔을 때 데이터의 분류를 비교적 정확하게 알려줌.
    - Clustering(Unsupervised Learning)
        - Given : 데이터 간의 유사도를 줌.
        - Find : 유사한 데이터 끼리 군집을 만듦.
        - Goal : 클러스터 셋을 만듦.
6. Evaluation / Interpretation
    - 만족스러운 결과인가?

