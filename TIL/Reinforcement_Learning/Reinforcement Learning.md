# Reinforcement_Learning

이 문서는 [Notion]()에서 Export한 md 파일입니다.

# Symbols

$S$ : set of States; 상태의 집합

$P_{tran}$ : state transition Probability; 다른 상태로 넘어갈 확률.

$R$ : Reward function; 보상 함수

$\gamma$ : Discounting factor; 미래의 가치를 현재로 환산할 때의 할인율.

$A$ : set of Action ; 행동의 집합.

$V(s)$ : State - value function; state 의 가치를 계산하는 함수

$Q_\pi(s,a)$: Action - value function; state에서 Action의 가치를 계산하는 함수.

$V^*(s) = \underset{\pi}{\max}V\pi(s)$ : optimal state value function; 상태의 기댓값을 최대로.

$Q^*(s,a) = \underset{\pi}{\max}Q_\pi(s,a)$: optimal action value function; 액션의 기댓값을 최대로

$\pi^*(a|s)$ : Optimal policy;  최대값을 가지는 Action Value function 을 선택하는 policy

$N(s)$ : state s에 방문한 횟수.

$\mu$ : Behavior Policy

$\hat{V}(s;w) \approx V_\pi(s) \text{ or } \hat{Q}(s,a;w) \approx Q_\pi(s,a)$ : 근사치  

$E(w) = \mathbb{E}[(V_\pi(s)-\hat{V}(s;w))^2]$ : Objective Function = Loss function

$A^{\pi_\theta}(s,a) = Q^{\pi_\theta}(s,a) - V^{\pi_\theta}(s)$ : Action value function. $Q$에서 Baseline $V$를 뺌.

$\delta^{\pi_\theta} = r + \gamma V^{\pi_\theta}(s') - V^{\pi_\theta}(s)$: Delta function. Action Value function에서 $Q$를 Bellman equation에 의해 $V$로 치환

---

# Bellman Equation

$$V^*(s) = \underset{a}{\max}Q^*(s,a)
\\
Q^*(s,a)= R^a_s + \gamma \underset{s' \in S}\sum  P^a_{ss'},V^*(s')
\\
V^*(s) = \underset{a}{max}(R^a_s + \gamma \underset{s' \in S}{\sum} P^a_{ss'},V^*(S') \\
Q^*(s,a) = R^a_s + \gamma \underset{s' \in S}{\sum}P^a_{ss'},(\underset{a}{max}Q^*(s',a'))$$

# Dynamic Programming

Planning: Model-base(Env를 알때) Optimal한 value와 policy를 찾는 문제

Prediction: 

- MDP와 $\pi$가 주어지고
- $\pi$를 따랐을때의 value function을 계산하여 policy의 성능 측정

Control: 

- MDP 주어짐.
- optimal한 value function과 optimal policy를 구함.

## Policy Iteration

반복적으로 State Value를 업데이트.

어느 순간 값이 수렴하게 됨.

Optimal Policy는 각 State에서 다음 state로 Greedy하게 진행하면 Optimal Policy가 나옴.

![Reinforcement_Learning/Untitled.png](Reinforcement_Learning/Untitled.png)

Random Policy → Optimal Policy (Evaluate → Improve → Evaluate) 반복.

![Reinforcement_Learning/Untitled%201.png](Reinforcement_Learning/Untitled%201.png)

## Value Iteration

State Value 업데이트, 업데이트를 통해 Optimal한 Policy가 나옴.

모든 액션을 수행 했을 때, max 값을 업데이트.

![Reinforcement_Learning/Untitled%202.png](Reinforcement_Learning/Untitled%202.png)

---

# Monte-Carlo(MC) Learning

- MC learns from complete episodes(Episodic)
    - Terminate state가 존재해야 함.
    - $V(S_t) = \mathbb{E} [G_t|S_t = s]$&
    - $G_t = r_t + \gamma r_{t+1} + \gamma^2r_{t+2} + \cdots + \gamma ^{T-1}r_{T}$
- $V(s) = Z(s) / N(s)$
- Zero bias / high variance
    - 초기값에 영향을 많이 받지 않음.
    - 

### Non-stationary problems

환경이 계속해서 변할 때, $\alpha$를 작은 수로 설정하여 과거의 episode들은 영향력이 작아지는 효과가 있다. (forget old episodes)

$$V(S_t) \leftarrow V(S_t) + \alpha(G_t-V(S_t))$$

# Temporal-Difference (TD) Learning

- incomplete episodes.
- Update value with estimated return. 예측된 return을 통해 value를 업데이트.

$$V(S_t) \leftarrow V(S_t) + \alpha(r_{t+1}+\gamma V(S_{t+1})-V(S_t))
\\ \color{red}
\textbf{TD target: } r_{t+1}+\gamma V(S_{t+1})
\\ \color{blue}
\textbf{TD Error } \delta_t \textbf{ : } 
r_{t+1}+\gamma V(S_{t+1})-V(S_t)$$

- final outcome이 나오지 않아도 업데이트.
- some bias / low variance
    - 초기값에 영향을 많이 받음.

---

# On-policy

### Model of MDP

Model을 알고 있을 땐, 다른 State로 가는 확률과 state value를 최대화 하는 policy를 설정.

$$⁍$$

### Model-free

Model을 모를땐, Q function으로 최대화하는 policy를 설정.

$$⁍$$

Q 값이 가장 높은 값만 선택하게 되는 issue가 생김.

→ $\epsilon$-Greedy Exploration 수행.

$\epsilon$(매우 낮은 확률)의 확률로 다른 무작위의 action을 수행.

**Every episode**

Policy evaluation : Monte-Carlo policy evaluation, $Q = q\pi$

Policy improvement : $\epsilon$-greedy policy improvement

$\epsilon \leftarrow 1/k$  결국 0으로 수렴함.

## Sarsa

![Reinforcement_Learning/Untitled%203.png](Reinforcement_Learning/Untitled%203.png)

$$Q(S,A) \leftarrow Q(S,A) + \alpha(R + \gamma Q(S',A') - Q(S,A)) \\
\text{TD Target : }R + \gamma Q(S',A')$$

모든 step마다 업데이트.

Estimated return. 예측 값으로 업데이트.

---

# Off-policy Learning

target policy와 behavior policy가 다름. (실제 수행하는 policy와 평가하는 policy가 다름)

Target Policy = greedy

Behavior Policy = $\epsilon$ - greedy

exploratory policy를 사용했을 때와 optimal policy를 사용했을 때를 비교하며 학습가능.

이전의 policy에서 사용했던 결과를 사용해서 학습이 가능.

### Importance Sampling

$$\begin{align*}
\mathbb{E}_{X \sim P}[f(X)] &= \sum P(X)f(X)\\
&= \sum Q(X) \frac{P(X)}{Q(X)} f(X)\\
&= \mathbb{E}_{X \sim \color{red}Q} \left[ \color{red}\frac{P(X)}{Q(X)} \color{black}f(X) \right]
\end{align*}$$

$\mu$ 를 이용하여 target policy인 $\pi$ 를 평가.

각 스텝에서 $\frac{\pi}{\mu}$ 를 반영하여, 실제로는 $\mu$ 를 따랐지만 $\pi$를 수행했을 때를 평가할 수 있음.

→ MC에선 variance가 너무 커져서 적용이 힘듦.

→ TD에서 적용

$$V(S_t) \leftarrow V(S_t) + \alpha(\color{red}\frac{\pi(A_t|S_t)}{\mu(A_t|S_t)}(R_{t+1}+\gamma V(S_{t+1}))\color{black}-V(S_t))$$

## Q-learning

행동은 exploration을 수행하지만, target은 greedy하게 평가.

$A_{t+1} \sim \mu(\cdot| S_t)$ : Behavior Policy | $\epsilon$ greedy

$A' \sim \pi(\cdot| S_t)$ : Target Policy | greedy 

$$Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha(R_{t+1} + \gamma Q(S_{t+1},\color{blue}A'\color{black}) - Q(S_t,A_t))$$

![Reinforcement_Learning/Untitled%204.png](Reinforcement_Learning/Untitled%204.png)

$$Q(S,A) \leftarrow Q(S,A) + \alpha(R + \gamma \underset{a'}{\max}Q(S',a') - Q(S,A))$$

- 다음 State의 action 중 가장 큰 값을 선택하여 업데이트.
- Policy evaluate 와 improvement 가 합쳐진 형태.

# Value Function Approximation

실제 문제에선 MDP가 굉장히 큼. 이 상태를 다 저장할 수 없음.

Continuous state space 에서도 사용.

Value Function을 Approximation 해야하는데, 이때 Deeplearning 방법론을 사용한다.

![Reinforcement_Learning/Untitled%205.png](Reinforcement_Learning/Untitled%205.png)

Policy Network를 구성.

State를 입력하고, Deep Learning 방법론을 이용하여 Value Function을 구성한다. 이후 결과값으로 나오는 $\hat{Q}$ 를 바탕으로 action을 선택. Policy를 생성.

## Gradient Descent for Value Function Approx.

$$⁍$$

Objective function 을 줄이는 방향으로 w를 업데이트.

![Reinforcement_Learning/Untitled%206.png](Reinforcement_Learning/Untitled%206.png)

$$w_1(i+1) = w_1(i) - \eta \nabla E(w_1(i))$$

![Reinforcement_Learning/Untitled%207.png](Reinforcement_Learning/Untitled%207.png)

- Gradient descent

    Objective Function을 편미분 한 공식. 

- Stochastic gradient descent

    회차가 반복 될 때마다 $w$를 업데이트.

## MC

$$\vartriangle w = \eta(G_t - \hat V (S_t;w)) \nabla _w \hat V(S_t;w)$$

$G_t$를 기준으로  $w$를 업데이트.

## TD

$$\vartriangle w = \eta(R_{t+1} + \gamma\hat V(S_{t+1};w) - \hat V (S_t;w)) \nabla _w \hat V(S_t;w)$$

$R_{t+1} + \gamma\hat V(S_{t+1};w)$  를 기준으로 $w$ 업데이트.

## Deep Q-Networks

- DQN use experience replay and fixed Q-target
    - Take action $a_t$ according to $\epsilon$ -greedy policy
    - Store transition $(s_t,a_t,r_{t+1},s_{t+1})$ in replay memory $D$
    - Compute Q-learning targets (fixed parameters $w^\bold{-}$)
        - Fixed $w^-$ 를 Iteration을 어느 정도 수행한 뒤 업데이트
    - Optimize MSE between Q-network and Q-learning targets

$$E(w) = \mathbb{E}_{s,a,r,s'~D}[(r + \gamma \underset{a'} \max Q(s',a';w^-) - Q(s,a;w))^2]$$

# Policy Gradient

## Advantages of Policy-Based RL

- Advantages:
    - Better convergence properties
    - Effective in high-dimensional or continuous action spaces
    - Can learn stochastic policies (Value Base는 Greedy해서 deterministic)
- Disadvantages:
    - Typically converge to a local rather than global optimum.
    - Evaluating a policy is typically inefficient and high variance.

![Reinforcement_Learning/Untitled%208.png](Reinforcement_Learning/Untitled%208.png)

## Actor-Critic

Policy $\pi _\theta$를 추정하기 위함. $\theta$ 값을 업데이트 해야 하는데 Policy Gradient를 사용하여 업데이트 한다.

$$\begin{align*}
\nabla_\theta J(\theta)
&= \mathbb{E}_{\pi _\theta}[\nabla_\theta \log \pi_\theta (s,a) G_t] &&\text{REINFORCEMENT}
\\
&= \mathbb{E}_{\pi _\theta}[\nabla_\theta \log \pi_\theta (s,a) Q_w(s,a)] &&\text{Q Actor-Critic}
\\
&= \mathbb{E}_{\pi _\theta}[\nabla_\theta \log \pi_\theta (s,a) A(s)] &&\text{Advantage Actor-Critic}
\\
&= \mathbb{E}_{\pi _\theta}[\nabla_\theta \log \pi_\theta (s,a) \delta_v] &&\text{TD Actor-Critic}
\end{align*}$$

### Q Actor-Critic

- $G_t$를 이용하는 MC는 variance가 크다.
- critic을 사용하여 action-value function을 업데이트 한다.
- Apporox된 Value값을 갖고 policy gradient도 계산.
- Critic = action value function을 업데이트 ($w$)
- Actor = $\theta$ 를 업데이트. 업데이트 시에 critic에서 나온 추정된 value function을 쓰자.

### Advantage Actor-Critic

- Return이 너무 커서 편향이 생길 수 있어서 Baseline을 이용해 편차를 줄임.

    $A^{\pi_\theta}(s,a) = Q^{\pi_\theta}(s,a) - V^{\pi_\theta}(s)$ : Action value function. $Q$에서 Baseline $V$를 뺌.

    하지만 Advantage Function을 이용하려면 $Q_w(s,a) = Q^{\pi_\theta}(s,a)$, $V_v(s) = V^{\pi_\theta}(s)$ 를 모두 추정해야함. ⇒ 계산이 복잡.

### TD Actor-Critic

- Q function을 V로 바꾸자. → Delta function

    $\delta^{\pi_\theta} = r + \gamma V^{\pi_\theta}(s') - V^{\pi_\theta}(s)$

    $\delta_v = r + \gamma V_v(s') - V_v(s)$