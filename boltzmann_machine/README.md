# Boltzmann Machine 

A repository **Normal (Classic) Boltzmann Machines (BM)** from scratch using NumPy, fully managed via the `uv` toolchain.

---

## 🧮 Mathematical Architecture

Boltzmann Machines are energy-based, generative, stochastic neural networks. The system defines a probability distribution over states by assigning lower energy values to configurations that resemble the training data.

### 1. The Normal (Classic) Boltzmann Machine
A Normal Boltzmann Machine is a recurrent network where **every unit is connected to every other unit**. There is no concept of clean, separate layers; instead, the state vector $s \in \{0, 1\}^N$ contains both visible ($v$) and hidden ($h$) units merged together ($s = [v, h]$).

#### The Energy Function
The joint energy of a state configuration vector $s$ is explicitly defined as:
$$E(s) = -\sum_{i=1}^{N} \theta_i s_i - \sum_{i < j} w_{ij} s_i s_j$$

Where:
* $w_{ij}$ is the symmetric weight between unit $i$ and unit $j$ ($w_{ij} = w_{ji}$).
* $w_{ii} = 0$ (no unit can connect to itself).
* $\theta_i$ is the bias vector for unit $i$.

#### Asynchronous Gibbs Sampling
Because intra-layer connections exist, units are not conditionally independent. Finding expectations requires updating units **one at a time** in an asynchronous sequence until the system reaches thermal equilibrium:
$$P(s_i = 1 \mid s_{-i}) = \sigma\left( \theta_i + \sum_{j \neq i} w_{ij} s_j \right)$$

Where $\sigma(x) = \frac{1}{1 + e^{-x}}$ is the logistic sigmoid function.

---

### 2. The Restricted Boltzmann Machine (RBM)
An RBM restricts the architecture to a bipartite graph: **visible units only connect to hidden units**, meaning there are absolutely no intra-layer connections. This structure allows for fast, parallelized matrix computations.

#### The Energy Function
The joint configuration energy of a visible layer vector $v \in \{0, 1\}^{V}$ and a hidden layer vector $h \in \{0, 1\}^{H}$ is:
$$E(v, h) = -\sum_{i=1}^{V} a_i v_i - \sum_{j=1}^{H} b_j h_j - \sum_{i=1}^{V}\sum_{j=1}^{H} v_i W_{ij} h_j$$

Or in compact matrix notation:
$$E(v, h) = -a^\top v - b^\top h - v^\top W h$$

Where $W \in \mathbb{R}^{V \times H}$ represents the weight matrix, and $a$ and $b$ are the visible and hidden bias vectors respectively.

#### Conditional Independence & Parallel Sampling
Because units within the same layer do not influence each other directly, the conditional probabilities factorize perfectly. This allows the entire layer to be sampled at once in parallel:
$$P(h_j = 1 \mid v) = \sigma\left(b_j + \sum_{i=1}^{V} v_i W_{ij}\right)$$
$$P(v_i = 1 \mid h) = \sigma\left(a_i + \sum_{j=1}^{H} W_{ij} h_j\right)$$

---

## 📉 Parameter Learning and Optimization

Both models are trained via gradient ascent on the log-likelihood of the training data. The gradient with respect to a weight component evaluates to:

$$\frac{\partial \ln P(v)}{\partial w_{ij}} = \langle s_i s_j \rangle_{\text{data}} - \langle s_i s_j \rangle_{\text{model}}$$

1. **$\langle s_i s_j \rangle_{\text{data}}$ (Positive Phase):** The correlation of units when the training data is actively clamped onto the visible units.
2. **$\langle s_i s_j \rangle_{\text{model}}$ (Negative Phase):** The structural expectation of unit correlations when the network runs completely unclamped (free-running). 

### Optimization Techniques
* **Normal BM:** Runs a long, asynchronous Gibbs chain over the entire network to approximate $\langle s_i s_j \rangle_{\text{model}}$, which is computationally heavy.
* **Restricted BM (Contrastive Divergence - $CD-k$):** Approximates the negative phase efficiently by initializing a Gibbs chain at the data vector $v_0$ and running parallel layer updates for only $k$ steps ($v_0 \to h_0 \to v_k \to h_k$).

Weight and bias adjustments follow standard gradient updates scaled by learning rate $\eta$:
$$\Delta w_{ij} = \eta \cdot \left( \langle s_i s_j \rangle_{\text{data}} - \langle s_i s_j \rangle_{\text{model}} \right)$$

---

## 🚀 Execution Instructions

Using `uv`, you don't need to manually create virtual environments or deal with local caching issues. Execute scripts directly within the environment:

```bash
# To run the training and validation loops
uv run train.py