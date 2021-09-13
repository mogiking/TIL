# Deep Learning

'21.2학기 딥러닝 수업의 강의 노트 입니다.

# 1강. 21.09.13.

## Perceptron
퍼셉트론.
초기화가 중요. Bias는 shift. w는 weight.

### Multi Layer Perceptron
FCN. Fully Connect Network(Layer).

### Sigmoid Neuron
Non linear.
RELU. 
확률 distribution 으로 나타내기 위해.

여러가지 activation function이 존재함.
주로 Relu를 사용. Dead relu가 발생하기도 함

### Feedforward Neural Network
과적합 되기 때문에 Drop Out을 사용해서 막음.
alpha=0.5 를 적용시 50%가 Drop out. 오버피팅을 막음.

AlexNet의 마지막 풀리커넥트 레이어가 대부분의 가중치를 갖고있음.
Input과 Output 이 고정이 되어있음.
학습을 240\*240으로 했는데 200\*200을 구할 수 없음.
때문에 FCN을 사용하여 구현.



## Deeplearning 학습 방법

### Stochastic Gradient Decent
경사하강법.
가장 최소/최대점을 찾음. loss function이 최소가 되는 지점을 찾음.
Local min에 빠지지 않게 여러가지 조치를 취한다.

### Backpropagation

Backpropagation(오차역전파)을 이용하면 복잡한 형태의 NN도 Chain Rule을 이용하여 쉽게 업데이트가 가능하다.
오차를 각 Node로 역으로 전달하여 가중치를 업데이트 할 수 있다.

1. Forward Pass 시에 Local Gradient를 저장한다.
2. Backward 시에 저장해둔 Local Gradient와 Gloabl Gradient를 곱하여 최종 미분 값을 얻는다.

- 내일 다시 업데이트