
# PINN & Function Approximation with PyTorch

This repository contains **PyTorch** implementations of Neural Networks designed for function approximation and **Physics-Informed Neural Networks (PINNs)** used to solve Ordinary Differential Equations (ODEs), specifically the Damped Harmonic Oscillator.

The code explores advanced training techniques such as mixed optimization strategies, hard constraint enforcement ("Forced CC"), and neuron saturation analysis.

## 👤 Author

**Marco Mas Sempre**


## 📂 Repository Contents

### 1. Basic Approximation

* **`NN_Function_Aproximator.py`**
  * A simple Feed-Forward Neural Network designed to approximate the function .
  * Serves as a baseline for understanding the network structure and training loop using the Adam optimizer.



### 2. Physics-Informed Neural Networks (PINNs)

* **`PINN_EDO_ForzedCC_MixMeth_NeuronAnimations.py`**
  * Solves the **Damped Harmonic Oscillator** equation. **Features:**
  1. **Hard Constraints (Forced CC):** The network ansatz is structured to mathematically satisfy initial conditions () automatically, removing the need for boundary loss terms.
  2. **Mixed Optimization:** Uses **Adam** for initial coarse training, followed by **LBFGS** for fine-tuning.
  3. **Visualization:** Includes code to animate the evolution of weights and biases during training.




* **`PINN_EDO_ForzedCC_MixMeth_NeuronColapse.py`**\n
   * Solves the **Damped Harmonic Oscillator** equation. **Features:**
  1. **Hard Constraints (Forced CC):** The network ansatz is structured to mathematically satisfy initial conditions () automatically, removing the need for boundary loss terms.
  2. **Mixed Optimization:** Uses **Adam** for initial coarse training, followed by **LBFGS** for fine-tuning.
  3. A **research** script that investigates **Neuron Collapse** (saturation) within the PINN.



## 🚀 Key Techniques

* **Hard Constraints (Ansatz):** Instead of soft constraints (MSE loss on boundaries), the solution is modeled as:
  $$x_{net}(t) = x_0 + v_0 t + NN(t) \cdot t^2$$. This ensures the initial conditions are satisfied exactly at $t= 0$.
* **Optimization Strategy:**
  1. **Adam:** Fast descent to avoid local minima .
  2. **LBFGS:** Quasi-Newton method for high-precision convergence once near the solution.


## 📋 Requirements
You need Python installed along with these libraries:

```bash
pip install torch torchvision torchaudio numpy matplotlib
```
