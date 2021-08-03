# Statistics

# Symbols

Intersection(교집합): $A \cap B = AB$ ( A and B )  

Union(합집합): $A \cup B$ ( A or B )

Complemnet(여집합): $A^c = S -A$ 

De Morgan's law : 

$$(E \cup F)^c = E^c \cap F^c \\
(E \cap F)^c = E^c \cup F^c$$

 Mutually Exclusive (상호 배타) : $A \cap B = AB = \phi$

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