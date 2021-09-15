# Probability

'21.2학기 수업.

## 수업의 목표

$P(x)$의 의미를 아는 것.

What is a probability distribution?

→ Probability events to happen

→ How frequent some events occur?

Ex) What is the probability of having number 1 from a fair dice?

What is a probability density function?

Ex) **Gaussian Distribution** :  $P(x)=\frac{1}{\sqrt{2\pi\sigma^2}}\exp[-\frac{1}{2\sigma^2}(x-\mu)^2]$

What is a random variable?

→ Gaussian random variable: a mapping between sample space and $\mathbb{R}$

## Set Theory

집합 이론. 집합과 확률은 무관해 보이지만 집합 이론을 먼저 배워야 다음을 이해할 수 있다.

$\mathbb{R}$:Real line(set of all real numbers)

$x \in \mathbb{R}$ : x belongs to R

- Set : Collection of object
- element : a member of a set
- subset : {a,b} is a subset of {a,b,c}
- universal set : {x,y,z} is a universal set of {x,y} and {y,z}
- Disjoint sets : $A \cap B = \emptyset$
- partition of A : A = {1,2,3,4} → partition of A = {{1},{2},{3,4}}

    각각 disjoint, 모두 합치면 원래의 set

- Cartesian Product:

    $A \times B = \{(a,b)|a \in A, b \in B \}$ , Ex) $\R^2$ 2차원 평면

- Power set : $2^A$ the set of all subsets of A, 갯수가 $2^A$개 이다.
- Cardinality : $|A|$, the number of elements of a set.

    → finite, infinite

    → countable, uncountable

    → 자연수의 집합 vs 실수의 집합, 어느 것이 더 클까?

    - Countable : There is a one-to-one correspondence between the set of all natural numbers.

        $|\Z| = N_0$ (aleph null)

    - Uncountable: One of well-known set is [0,1]

        $|\R| = 2^{N_0}$ (continuum)

- Function of Mapping :

     $f: U \to V \text{ is for set}\\ f:u\mapsto v \text{ is for elements}$

    $\sin(x)$:

    - Domain = $\R$
    - codomain = $\R$
    - Range = [-1,+1]
- inverse image

    $f^{-1}(B) = \{x\in U | f(x) \in B\}, \text{where }B \subset V$