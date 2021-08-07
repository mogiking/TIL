# Statistics

# Symbols

Intersection(교집합): $A \cap B = AB$ ( A and B )  

Union(합집합): $A \cup B$ ( A or B )

Complemnet(여집합): $A^c = S -A$ 

De Morgan's law : 

$$(E \cup F)^c = E^c \cap F^c \\
(E \cap F)^c = E^c \cup F^c$$

 Mutually Exclusive (상호 배타) : $A \cap B = AB = \phi$

조건부 확률 : $P(E|F) = \frac{P(EF)}{P(F)}$

# 1강.

## Permutations

- A permutation is an arrangement of all or part of a set of object. Thus, the **order of things** is considered.
- In other words, the number of ways of ordering a set of objects assume that the **objects are distinguishable**.

$$n! \\ \rightarrow 0! = 1$$

### Nondistinguishable Permutations

e.g) 10 player ; 1 freshman , 2 sophomores, 4 juniors, 3 seniors

$$\frac{10!}{1!\cdot2!\cdot3!\cdot4!}=12,600$$

## Combinations

- A combination of r object from n objects (n ≥ r) is  an unordered collection of r objects selected without replacement from the group of n object
- $\binom{n}{r}$;n combination r ; n choose r

### The Binomial Theorem (이항정리)

$$(x+y)^n = \sum \binom{n}{i}x^iy^{n-i}

\\ \\
\text{Example}:
(x + y)^3 = \binom{3}{0}x^0y^3 + \binom{3}{1}x^1y^2 + \binom{3}{2}x^2y^1 + \binom{3}{3}x^3y^1 $$

# 2강. Axioms of Probability(확률의 공리)

Axiom: 증명을 할 필요가 없는 Fact. 

1. $0 \leq P(E) \leq 1$
2. $P(S) = 1$
3. If $E_1,E_2, \cdots$ is a sequence of events with $E_i \cap E_j = \phi$ for all $i$≠ $j$ , 

    then $P(U_{i=1}^\infin E_i) = \sum_{i=1}^\infin P(E_i)$

    상호 배타적인 이벤트들의 합집합의 확률은 각 Event들의 확률을 더한 것과 같다. 

    상호 배타적인 이벤트들이 적어도 하나의 이벤트가 발생할 확률은 각각의 확률의 합으로 계산한다.

**Proposition 1**: $P(E^c) = 1 - P(E)$

**Proposition 2**: **If $E \subset F$, then** $P(E) \leq P(F)$

**Proposition 3(*inclusion-exclusion* identity)**: $P(E \cup F) = P(E) + P(F) - P(EF)$

## Equally Likely Outcomes

동일 확률. e.g. 주사위 던지기.  $S = \{ 1,2,3, \cdots , N\}$

$$P(\{1\}) = P(\{2\}) = \cdots = P(\{N\}) \\
\text{For any event }E,\ P(E) = \frac{number\ of\ outcomes\ in\ E}{number\ of\ outcomes\ in\ S}$$

## 용어정리

### Sample Space( 표본 공간 )

the sample space $S$ is a set that contains all possible experimental outcomes. 실험으로부터 나온 **모든** 결과를 담고 있는 **집합**

### Experiment(실험)

데이터를 생성하는 모든 과정.

### Event(사건)

A subset of Sample space. 표본 공간.

$$x \in S \\
x \in E \text{  then  } x \in S $$

### Discrete Sample Space (이산형 표본공간)

- Tossing a coin: $S = \{H,T\}$
- Tossing a dice: $S = \{1,2,3,4,5,6\}$
- Count number of people that enter the post office $S = \{0,1,2,3,4,\cdots \}$

### Continuous Sample Space (연속형 표본 공간)

- Values between 0 and 130:  $S = \{x | 0 < x < 130\}$
- Points inside a circle of radius2: $S = \{(x,y) | x^2 + y^2 < 4\}$

# 3강. Conditional Probability(조건부확률)

The probability that event $E$ occurs given that event $F$ occurs

$$P(E|F) = \frac{P(EF)}{P(F)}\text{ for } P(F) > 0$$

- Probability of $E$ given $F$
- Probability of $E$ conditional on $F$

$E$ and $F$ are mutually exclusive : 0

$F \subset E$ : 1

### Multiplicative Rules

$$P(AB) = P(A)P(B|A) = P(B)P(A|B)$$

![Statistics%206d9a192cee654445af25736311443cdd/Untitled.png](Statistics%206d9a192cee654445af25736311443cdd/Untitled.png)

## Bayes' Rule (베이즈 룰)

- Rule that **calculates the posterior probability** based on **prior and data probabilities**
- Posterior probability can be considered as an updated version of the prior probability

Let $A_1$, $A_2$, and $A_3$ be a partition of the sample S and they are mutually exclusive. B is an event.

![Statistics%206d9a192cee654445af25736311443cdd/Untitled%201.png](Statistics%206d9a192cee654445af25736311443cdd/Untitled%201.png)

**Law of Total Probability**

$$\begin{align*}
P(B) &= P(A_1B) + P(A_2B) + P(A_3B) \\ &=\sum_{i=1}^3 P(A_i)P(B|A_i)
\end{align*}$$

Suppose we know the following probabilities:

$P(A_1),P(A_2),P(A_3)$ : Prior probability (사전확률| 정보$(B)$를 받기 전의 확률)

$P(B|A_1),P(B|A_2),P(B|A_3)$ : Data probability (사전정보 조건부확률)

We would like to know the following probabilities:

$P(A_1|B),P(A_2|B),P(A_3|B)$ : Posterior probability (사후확률| 정보$(B)$로 업데이트 된 확률)

$$\begin{align*}
P(A_1|B) &= \frac{P(A_1)P(B|A_1)}{P(B)} \\&= \frac{P(A_1)P(B|A_1)}{P(A_1)P(B|A_1)+P(A_2)P(B|A_2)+P(A_3)P(B|A_3)}
\end{align*}$$

## Odds (아즈, 오즈)

$$\frac{P(A)}{P(A^c)} = \frac{P(A)}{1-P(A)}$$

- 성공 확률이 1일 경우 Odds는 무한대 값.
- 성공 확률이 0일 경우 Odds는 0

**배당에서 사용될 경우**

각 국가의 우승 Odds:

- 독일 4/1         P(독일) = 1/5
- 한국 500/1     P(한국) = 1/500