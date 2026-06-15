## Improving almost colorings


## Algorithm 1: Automated almost-coloring formalization

In our prior work, we used the following algorithm:

1. **Initial training.** Train $p_\theta : \mathbb{R}^2 \to \Delta_{c+1}$ to minimize Equation (5) on a large enough box $[-R, R]^2$.
2. **Periodicity extraction.** Determine two vectors $v_1, v_2 \in \mathbb{R}^2$ with $0 \ll \angle(v_1, v_2) \ll \pi$ such that the coloring (largely) consists of tiling the parallelogram
$$\mathcal{P} = \{\alpha v_1 + \beta v_2 : \alpha, \beta \in [0, 1)\}$$
along the lattice $\Lambda = \{n_1 v_1 + n_2 v_2 : n_1, n_2 \in \mathbb{Z}\}$.
3. **Periodicity-constrained retraining.** Form the invertible change-of-basis matrix $M = [v_1 \ v_2] \in \mathbb{R}^{2 \times 2}$. Prepend the mapping $x \mapsto M^{-1}x \pmod 1$ to $p_\theta$, which enforces exact periodicity over $\Lambda$, and retrain.
4. **Discrete almost-coloring.** Discretize $\mathcal{P}$ into $kl$ copies of $\{\alpha v_1 / k + \beta v_2 / l : \alpha, \beta \in [0, 1)\}$ and determine a color for each parallelogram pixel by sampling $p_\theta$ at its respective center.
5. **Iteratively fix remaining conflicts.** Determine a discrete mask in which conflicts need to be avoided around each parallelogram pixel to obtain a formal coloring. Iteratively reduce any remaining conflicts by solving an auxiliary minimum edge cover problem and recoloring some parallelograms. After a fixed number of rounds, resolve any remaining conflicts by recoloring with the additional color $c + 1$.

Here, we want to skip step 1 and 2 and directly start with step 3. The vectors themselves are implemented as trainable parameters as of now so let's just play around with different values. We want to focus our research mainly on **step 5**. After obtaining a discrete (constant on parallelogram pixels) coloring, it is not exactly clear 

---

## Results Table

| # colors | 1 | 2 | 3 | 4 | 5 | 6 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **best known** | 77.04% | 54.13% | 31.20% | 8.25% | 3.74% | 0.0149% |

## Ideas

* The hyperparameters in the training stage were not really tuned for this problem. We think that small MLPs (2-4 hidden layers, 32 - 128 hidden units per layer) with `sin` activation and siren initialization work reasonably well here but that is not settled. Same goes for learning rate schedule, weight decay etc. 
* The formulation with the lagrange weight is a bit ugly because it's a pain to tune. Here it could be cool to find some bilevel strategy to optimize the lagrange weight.
* Making the parallelogram trainable in the same way as the NN parameters is probably suboptimal. We should experiment with at least a different learning rate but probably even a strategy where we freeze the parallelogram for some steps and then unfreeze it (maybe even sometimes freeze the NN parameters for some steps and train only the parallelogram).
* Definitely pursue LP formulations of step 5 in the algorithm up there. Writing the LP down is easy (minimize number of bonus pixels such that no cells at distance approximate 1 have the same color, routines to get the cells at distance approximate 1 are in the verify_parallelogram.py files). But already for moderate discretizations solution might get a bit ugly, so think of routines. Definitely experiment with solving it with ipopt, simplex, other nice LP ormulations.

## Tasks

* [] A "success" would be an almost 5 coloring with less than 3.74% of pixels covered using the "bonus" color. 
* [] Once all of that is done we apply it to 3D. 