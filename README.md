# Restricted Boltzmann Machine (RBM) from Scratch

A clean, high-performance implementation of a binary Restricted Boltzmann Machine (RBM) written using NumPy and managed via `uv`. 

An RBM is a generative, stochastic, two-layer neural network designed to learn a probability distribution over its inputs. "Restricted" implies that there are **no intra-layer connections**—visible units only connect to hidden units, and vice versa, creating a bipartite graph.

---

## 🧮 Mathematical Architecture

### 1. The Energy Function
Because an RBM is an energy-based model, the joint configuration of the visible layers $v \in \{0, 1\}^{V}$ and hidden layers $h \in \{0, 1\}^{H}$ is assigned an overall scalar energy value. The lower the energy, the more compatible (likely) the configuration.

The energy of a state $(v, h)$ is explicitly defined as:

$$E(v, h) = -\sum_{i=1}^{V} a_i v_i - \sum_{j=1}^{H} b_j h_j - \sum_{i=1}^{V}\sum_{j=1}^{H} v_i W_{ij} h_j$$

Or in vector-matrix notation:
$$E(v, h) = -a^\top v - b^\top h - v^\top W h$$

Where:
* $W \in \mathbb{R}^{V \times H}$ represents the weight matrix between units.
* $a \in \mathbb{R}^{V}$ is the bias vector for the visible layer.
* $b \in \mathbb{R}^{H}$ is the bias vector for the hidden layer.

### 2. Joint and Marginal Probabilities
The network assigns a probability to every joint configuration $(v,h)$ via the Boltzmann distribution:

$$P(v, h) = \frac{1}{Z} e^{-E(v, h)}$$

Where $Z$ is the partition function (a normalizing constant summing over all possible combinations of states):
$$Z = \sum_{v} \sum_{h} e^{-E(v, h)}$$

The probability that the network assigns to a visible data vector $v$ (the marginal probability) is found by summing over all hidden states:
$$P(v) = \frac{1}{Z} \sum_{h} e^{-E(v, h)}$$

### 3. Conditional Probability & Layer Independence
Because there are no connections within the same layer, the hidden units are conditionally independent given the visible units. Thus, the conditional probability factorizes cleanly:

$$P(h|v) = \prod_{j=1}^{H} P(h_j|v)$$

Using the logistic sigmoid function $\sigma(x) = \frac{1}{1 + e^{-x}}$, the probability that the $j$-th hidden unit is activated ($h_j = 1$) given a visible vector $v$ is:
$$P(h_j = 1 \mid v) = \sigma\left(b_j + \sum_{i=1}^{V} v_i W_{ij}\right)$$

Conversely, given a hidden state vector $h$, the visible units are also conditionally independent:
$$P(v_i = 1 \mid h) = \sigma\left(a_i + \sum_{j=1}^{H} W_{ij} h_j\right)$$

---

## 📉 Learning via Contrastive Divergence ($CD-k$)

To train the RBM, we perform gradient ascent on the log-likelihood of the training data $\mathcal{L} = \sum_{v \in \mathcal{D}} \ln P(v)$. The derivative of the log-likelihood with respect to a weight $W_{ij}$ yields two distinct phases:

$$\frac{\partial \ln P(v)}{\partial W_{ij}} = \langle v_i h_j \rangle_{\text{data}} - \langle v_i h_j \rangle_{\text{model}}$$

* **$\langle v_i h_j \rangle_{\text{data}}$ (Positive Phase):** The expectation of the feature combination under the empirical data distribution. This is easy to compute directly using $P(h|v_0)$.
* **$\langle v_i h_j \rangle_{\text{model}}$ (Negative Phase):** The expectation under the model's full distribution. This requires calculating the partition function $Z$, which is exponentially difficult ($O(2^{V+H})$).

### Gibbs Sampling Approximation ($CD-k$)
To circumvent calculating $Z$, we use **Contrastive Divergence**. We approximate the model distribution by running a Gibbs chain initialized at the data sample $v_0$ for $k$ steps:

$$v_0 \longrightarrow h_0 \sim P(h|v_0) \longrightarrow v_1 \sim P(v|h_0) \longrightarrow h_1 \sim P(h|v_1) \dots \longrightarrow v_k$$

Using the states from $CD-1$ (where $k=1$), the parameter updates scaled by a learning rate $\eta$ are:

$$\Delta W = \eta \cdot \left( v_0^\top P(h|v_0) - v_1^\top P(h|v_1) \right)$$
$$\Delta a = \eta \cdot (v_0 - v_1)$$
$$\Delta b = \eta \cdot \left( P(h|v_0) - P(h|v_1) \right)$$

---

## 🚀 Execution Instructions

Using `uv`, you don't need to manually source a virtual environment. You can execute the training loop directly through the tool:

```bash
uv run train.py