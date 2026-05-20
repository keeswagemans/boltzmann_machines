# Research Index

This directory serves as the research and development hub for this project. It contains scratch implementations of various Boltzmann Machine architectures from first principles, alongside the core physics-informed hypothesis and benchmark scripts.

## Directory Structure & Implementations

### 🔬 [Research Hypothesis](file:///usr/local/google/home/wagemans/Documents/Github-Kees/boltzmann_machines/research/research_hypothesis)
This is the central experimental workspace where we integrate physical constraints into Boltzmann Machines.
* [pi_rbm.py](file:///usr/local/google/home/wagemans/Documents/Github-Kees/boltzmann_machines/research/research_hypothesis/pi_rbm.py): Physics-Informed Restricted Boltzmann Machine.
* [pi_bm.py](file:///usr/local/google/home/wagemans/Documents/Github-Kees/boltzmann_machines/research/research_hypothesis/pi_bm.py): Physics-Informed Classic Boltzmann Machine (Gaussian visible units).
* [pi_crbm.py](file:///usr/local/google/home/wagemans/Documents/Github-Kees/boltzmann_machines/research/research_hypothesis/pi_crbm.py): Physics-Informed Convolutional RBM (2D physical grids).
* [pi_dbn.py](file:///usr/local/google/home/wagemans/Documents/Github-Kees/boltzmann_machines/research/research_hypothesis/pi_dbn.py): Physics-Informed Deep Belief Network with stacked backpropagation.
* [pi_dbm.py](file:///usr/local/google/home/wagemans/Documents/Github-Kees/boltzmann_machines/research/research_hypothesis/pi_dbm.py): Physics-Informed Deep Boltzmann Machine (iterative mean-field).
* [evaluate.py](file:///usr/local/google/home/wagemans/Documents/Github-Kees/boltzmann_machines/research/research_hypothesis/evaluate.py): Complete benchmark script evaluating all models against a standard PINN baseline.
* [experiment_suite.py](file:///usr/local/google/home/wagemans/Documents/Github-Kees/boltzmann_machines/research/research_hypothesis/experiment_suite.py): Hyperparameter sweeps and noise sensitivity ablation studies.

### 🧠 Scratch Implementations (Binary / Unconstrained)
These directories contain scratch-built, pure-Python/NumPy reference implementations of the classic models:
* **[boltzmann_machine/](file:///usr/local/google/home/wagemans/Documents/Github-Kees/boltzmann_machines/research/boltzmann_machine)**: Reference Classic Boltzmann Machine with symmetric recurrent connections.
* **[restricted_boltzmann/](file:///usr/local/google/home/wagemans/Documents/Github-Kees/boltzmann_machines/research/restricted_boltzmann)**: Reference Restricted Boltzmann Machine using Contrastive Divergence (CD-k).
* **[convolutional_machine/](file:///usr/local/google/home/wagemans/Documents/Github-Kees/boltzmann_machines/research/convolutional_machine)**: Reference Convolutional RBM with 2D weight sharing.
* **[deep_belief_network/](file:///usr/local/google/home/wagemans/Documents/Github-Kees/boltzmann_machines/research/deep_belief_network)**: Reference Deep Belief Network stack.
* **[deep_boltzmann_machine/](file:///usr/local/google/home/wagemans/Documents/Github-Kees/boltzmann_machines/research/deep_boltzmann_machine)**: Reference Deep Boltzmann Machine with two hidden layers.
