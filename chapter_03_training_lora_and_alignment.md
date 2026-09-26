# CHAPTER 3: Adaptation & Optimization — Loss Functions, Backpropagation, LoRA/QLoRA & Preference Alignment

## 3.1 Architectural Overview & Layer Scope

In Chapters 1 and 2, we examined how tokens are mapped to high-dimensional latent space and routed dynamically through Self-Attention layers. However, an untrained model contains purely random projection weights. 

This chapter deconstructs how neural weights are updated, why full fine-tuning crashes production infrastructure, how **Low-Rank Adaptation (LoRA/QLoRA)** solves the memory wall, and how models are aligned with human intent via **DPO** and **RLHF**.

```text
               [ Foundation Model Weights (W_0) ]
               (Pre-trained on Trillions of Tokens)
                                │
   ┌────────────────────────────┴────────────────────────────┐
   ▼                                                         ▼
[ Full Parameter Fine-Tuning ]                            [ PEFT / LoRA ]
* Updates 100% of parameters                              - Freezes W_0 completely
* Requires 16–20 bytes/param VRAM                         - Injects low-rank adapters (B × A)
* High risk of Catastrophic Forgetting                    - Requires < 20% VRAM of full tuning
   │                                                         │
   └────────────────────────────┬────────────────────────────┘
                                ▼
                 [ Supervised Fine-Tuning (SFT) ]
                 (Format enforcement, JSON schema, Task data)
                                │
                                ▼
                 [ Preference Alignment Layer ]
                 ┌──────────────┴──────────────┐
                 ▼                             ▼
           [ RLHF + PPO ]                     [ DPO ]
     (Separate Reward Model)      (Direct Implicit Optimization)
```

---

## 3.2 The Loss Objective: Autoregressive Next-Token Prediction

Every generative LLM is trained on a single unified objective: **predicting the immediate next token $t_{i}$ given all preceding tokens $t_{<i}$**.

### 3.2.1 From Output Logits to Probabilities: The Softmax Layer
At the final layer of the network, the hidden state vector $\vec{h}_{\text{last}} \in \mathbb{R}^{d_{\text{model}}}$ is projected back to vocabulary dimensions $\vert{}V\vert{}$ via the Unembedding Matrix $W_U \in \mathbb{R}^{d_{\text{model}} \times \vert{}V\vert{}}$:

$$\vec{z} = \vec{h}_{\text{last}} W_U \quad \in \mathbb{R}^{\vert{}V\vert{}}$$

The resulting vector $\vec{z}$ contains unnormalized scalar scores called **logits**. To convert logits into a valid probability distribution where all values sit in $(0, 1)$ and sum to $1.0$:

$$P(t_i = k \mid t_{<i}) = \frac{e^{z_k}}{\sum_{j=1}^{\vert{}V\vert{}} e^{z_j}}$$

### 3.2.2 Cross-Entropy Loss (Negative Log-Likelihood)
If the true target token in the training text is $y$ (where index $y$ has ground-truth probability $1.0$ and all other tokens have $0.0$), the **Cross-Entropy Loss ($\mathcal{L}$)** penalizes the model based strictly on how much probability mass it allocated to that correct target:

$$\mathcal{L}_{\text{CE}} = - \log P(t_i = y \mid t_{<i}) = - z_y + \log\left(\sum_{j=1}^{\vert{}V\vert{}} e^{z_j}\right)$$

```text
Target Token: "Audit" (Index 3)

Case A (Well-trained Model):
P("Audit") = 0.85  ──>  Loss = -ln(0.85) = 0.162  (Minimal Penalty)

Case B (Confused/Untrained Model):
P("Audit") = 0.02  ──>  Loss = -ln(0.02) = 3.912  (Severe Penalty)
```

### 3.2.3 Perplexity: The Data Metric for Quality
In model evaluation and data filtering, raw loss is exponentiated to calculate **Perplexity ($PPL$)**:
$$PPL = e^{\mathcal{L}_{\text{CE}}}$$
- **Intuition:** Perplexity represents the effective number of tokens the model is blindly guessing between at each step. 
- A perplexity of $PPL = 10$ means the model is as uncertain as if it were choosing uniformly among 10 candidate words. Lower perplexity indicates superior predictive confidence.

---

## 3.3 Backpropagation & The Calculus of Weight Updates

Once loss $\mathcal{L}$ is computed, how do we adjust billions of weight matrices?

### 3.3.1 The Chain Rule: Attributing Error Across Layers
By applying the multivariate **Chain Rule** of calculus, the training engine computes the partial derivative of the loss with respect to every individual parameter $w_{ij}$ in the network:

$$\frac{\partial \mathcal{L}}{\partial W^{(l)}} = \frac{\partial \mathcal{L}}{\partial \vec{h}^{(l)}} \times \frac{\partial \vec{h}^{(l)}}{\partial \vec{z}^{(l)}} \times \frac{\partial \vec{z}^{(l)}}{\partial W^{(l)}}$$

- In relational terms, this is an **error-attribution audit log**: it traces backward from the final prediction error through each matrix multiplication to determine exactly which parameter contributed to the deviation.

### 3.3.2 The Optimizer: Stochastic Gradient Descent vs. AdamW
The baseline update rule is Gradient Descent:
$$W_{t+1} = W_t - \eta \nabla_{W} \mathcal{L}$$
*(where $\eta$ is the learning rate).*

However, standard SGD oscillates violently across steep loss ravines. Modern LLM training universally uses **AdamW (Adaptive Moment Estimation with Decoupled Weight Decay)**.

AdamW maintains two statistical tracking states for **every single weight parameter**:
1. **First Moment ($m_t$):** Moving average of gradients (momentum; tracks direction):
   $$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
2. **Second Moment ($v_t$):** Moving average of squared gradients (variance; scales step size per weight):
   $$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

#### The Parameter Update:
$$W_{t+1} = W_t - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda W_t \right)$$
*(where $\lambda W_t$ is decoupled weight decay to prevent parameters from exploding).*

---

## 3.4 The Memory Wall of Full Fine-Tuning

Why can you not simply run full backpropagation on an open-source 70B model using commercial GPUs?

### Memory Allocation per Parameter in Full 16-Bit Training:

| State Component | Precision | Bytes Allocated Per Parameter |
|---|---|---|
| **Model Weights ($W$)** | `FP16` or `BF16` | **2 Bytes** |
| **Gradients ($\nabla W$)** | `FP16` or `BF16` | **2 Bytes** |
| **AdamW Optimizer State: Momentum ($m_t$)** | `FP32` | **4 Bytes** |
| **AdamW Optimizer State: Variance ($v_t$)** | `FP32` | **4 Bytes** |
| **FP32 Master Weight Copy** (to avoid precision loss) | `FP32` | **4 Bytes** |
| **Total Static Memory Overhead** | — | **16 Bytes per Parameter** |

### The 70-Billion Parameter Math:
$$\text{Static VRAM} = 70 \times 10^9 \text{ parameters} \times 16 \text{ Bytes} = \mathbf{1,120 \text{ GB of VRAM}}$$

Adding dynamic memory for activations ($\approx 200 \text{ GB}$), full fine-tuning of a 70B model requires **at least sixteen 80GB H100 GPUs ($\approx \$480,000$ in hardware)**.

---

## 3.5 Parameter-Efficient Fine-Tuning (PEFT) & LoRA

To avoid updating all parameters, Edward Hu et al. (2021) introduced **LoRA (Low-Rank Adaptation)**, grounded in the **Intrinsic Rank Hypothesis**:
> *The weight updates $\Delta W$ during domain adaptation have a very low "intrinsic dimension" or rank compared to the massive original parameter space.*

### 3.5.1 The Mathematical Formulation of LoRA
Instead of modifying the pre-trained weight matrix $W_0 \in \mathbb{R}^{d \times k}$, LoRA **freezes $W_0$ entirely** and decomposes the update matrix $\Delta W$ into two low-rank matrices $A$ and $B$:

$$W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} (B \times A)$$

Where:
- $W_0 \in \mathbb{R}^{d \times k}$ (Frozen, no gradients computed)
- $B \in \mathbb{R}^{d \times r}$ (Trainable low-rank matrix)
- $A \in \mathbb{R}^{r \times k}$ (Trainable low-rank matrix)
- $r$: The rank hyperparameter, where $r \ll \min(d, k)$ (typically $r \in [8, 16, 32, 64]$)
- $\alpha$: Scaling constant (allows tuning adaptation strength without altering the learning rate)

```text
             Input Vector x (Dimension: 4096)
                           │
           ┌───────────────┴───────────────┐
           │                               │
           ▼                               ▼
   Frozen Weight W_0               Adapter Matrix A
    [ 4096 × 4096 ]                 [ 8 × 4096 ]
   (No Gradients!)                         │
           │                               ▼
           │                       Adapter Matrix B
           │                        [ 4096 × 8 ]
           │                               │
           │                               ▼
           │                     Scale by factor (α / r)
           │                               │
           └───────────────┬───────────────┘
                           ▼
                    Sum Operations (+)
                           │
                           ▼
                        Output
```

### 3.5.2 Parameter Footprint Comparison
Consider a single projection layer in a model where $d = k = 4,096$:
- **Original $W_0$ Parameters:** $4096 \times 4096 = \mathbf{16,777,216}$
- **LoRA Adapters ($r = 8$):**
  - Matrix $A$: $8 \times 4096 = 32,768$
  - Matrix $B$: $4096 \times 8 = 32,768$
  - Total Trainable Parameters: $32,768 + 32,768 = \mathbf{65,536}$
  - **Reduction:** **99.61% reduction in trainable parameters** for this matrix.

### 3.5.3 Zero-Initialization Invariance
To ensure the model output is completely identical to the base pre-trained model at step $t = 0$:
- Matrix $A$ is initialized using a Gaussian normal distribution $\mathcal{N}(0, \sigma^2)$.
- Matrix $B$ is initialized **strictly to zeros ($0$)**.

$$\Delta W = B \times A = 0 \times A = 0 \quad (\text{at initialization})$$
The model starts exactly at baseline performance and gradually learns domain-specific deltas without experiencing sudden initial divergence.

### 3.5.4 Zero Inference Latency (Weight Merging)
During deployment, you do not need to execute two separate matrix paths. Because matrix addition is associative, you can permanently merge the learned adapters back into the original weights:
$$W_{\text{deployment}} = W_0 + \frac{\alpha}{r} (B \times A)$$
The merged model has the **exact same architecture, size, and inference latency** as the original base model.

---

## 3.6 QLoRA: Quantized 4-Bit Adaptation

While LoRA reduces optimizer memory, the frozen base weights $W_0$ still consume 16-bit VRAM. **QLoRA (Dettmers et al., 2023)** enables fine-tuning by compressing base weights to 4-bit precision through three algorithmic innovations:

```text
┌─────────────────────────────────────────────────────────────┐
│                      QLoRA INNOVATIONS                      │
├─────────────────────────────────────────────────────────────┤
│ 1. NF4 Data Type: Quantizes weights to normal distribution  │
│ 2. Double Quantization: Quantizes the scale factors         │
│ 3. Paged Optimizers: Prevents OOM by spilling VRAM to RAM   │
└─────────────────────────────────────────────────────────────┘
```

### 1. NormalFloat4 (NF4) Quantization
Weights in neural networks naturally follow a zero-mean Gaussian distribution $\mathcal{N}(0, \sigma^2)$. Standard integer quantization (INT4) spaces quantization bins linearly, which wastes precision in the tails. NF4 constructs an information-theoretically optimal quantile grid where each 4-bit bin contains an equal number of expected parameters.

### 2. Double Quantization (DQ)
Quantization requires saving scaling constants ($c_1$) to convert 4-bit integers back to floats:
- Standard quantization adds an overhead of $32 \text{ bits} / 64 \text{ blocks} = 0.5 \text{ bits per parameter}$.
- Double Quantization quantizes the scaling constants themselves into 8-bit integers, reducing the footprint to **$0.127 \text{ bits per parameter}$** (saving $\approx 3 \text{ GB}$ of VRAM on a 70B model).

### 3. Paged Optimizers
Uses CUDA Unified Memory to automatically page optimizer states between GPU VRAM and physical system CPU RAM during memory-intensive sequence peaks, eliminating sudden out-of-memory crashes.

---

## 3.7 Alignment: Teaching Tone, Logic, and Safety

Supervised Fine-Tuning (SFT) teaches format and syntax, but suffers from two limitations:
1. **Sycophancy & Hallucination:** Models learn to mimic stylistic patterns in training text rather than verifying factual truth.
2. **Lack of Negative Feedback:** Cross-entropy only rewards positive tokens; it cannot effectively teach what *not* to say.

To address this, models undergo **Preference Alignment**.

```text
              Prompt: "How do I audit this offshore transaction?"
                                       │
                 ┌─────────────────────┴─────────────────────┐
                 ▼                                           ▼
         Output A (Helpful)                          Output B (Dangerous)
[Clear regulatory framework]               [Instructions on evading audits]
                 │                                           │
                 └─────────────────────┬─────────────────────┘
                                       ▼
                         Human / AI Preference Label
                              (Output A > Output B)
                                       │
                 ┌─────────────────────┴─────────────────────┐
                 ▼                                           ▼
      [ RLHF Architecture ]                        [ DPO Architecture ]
1. Train Reward Model (Scalar)               1. Mathematical substitution
2. Optimize policy via PPO actor-critic         into closed-form loss
3. Enforce KL-divergence penalty             2. Updates policy directly
```

### 3.7.1 Reinforcement Learning from Human Feedback (RLHF)
RLHF uses a multi-step pipeline:
1. Generate paired responses $(y_w, y_l)$ where $y_w$ is preferred (winning) and $y_l$ is dispreferred (losing).
2. Train a separate **Reward Model $r_\psi(x, y)$** using a pairwise ranking loss:
   $$\mathcal{L}_{\text{Reward}} = - \log \sigma\left(r_\psi(x, y_w) - r_\psi(x, y_l)\right)$$
3. Optimize the generation policy $\pi_\theta$ using **PPO (Proximal Policy Optimization)** against the reward model, adding a Kullback-Leibler (KL) divergence penalty to prevent the model from drifting too far from the original base model:
   $$\max_\theta \mathbb{E}\left[ r_\psi(x, y) - \beta \mathbb{D}_{\text{KL}}(\pi_\theta(y \mid x) \parallel \pi_{\text{ref}}(y \mid x)) \right]$$

### 3.7.2 Direct Preference Optimization (DPO)
RLHF via PPO is notoriously unstable: it requires keeping 4 separate neural networks in VRAM simultaneously (Actor, Critic, Reference Model, Reward Model).

**DPO (Rafailov et al., 2023)** proves mathematically that the objective can be solved **without training a reward model or using reinforcement learning**.

By expressing the ground-truth reward directly as a function of the optimal policy:
$$r(x, y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}$$

DPO derives a single, closed-form loss function that optimizes the language model directly on preference pairs:

$$\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = - \mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]$$

- **The Data Mechanism:** If the model increases the relative probability of the preferred answer $y_w$ faster than the reference model while penalizing $y_l$, the loss drops to zero. No reinforcement learning required.

---

## 3.8 Concrete Mathematical Walkthrough: LoRA Forward Pass

Let us compute an explicit forward pass for a single token through a LoRA-adapted layer.

### System Parameters:
- Input vector: $\vec{x} = [1.0, 2.0]$ ($d = 2$)
- Output dimension: $k = 2$
- Low-Rank Hyperparameters: Rank $r = 1$, Scaling Alpha $\alpha = 2.0$
- Scaling factor: $\frac{\alpha}{r} = \frac{2.0}{1} = 2.0$

### 1. Base Frozen Matrix ($W_0 \in \mathbb{R}^{2 \times 2}$):
$$W_0 = \begin{bmatrix} 0.5 & -0.5 \\ 1.0 & 0.0 \end{bmatrix}$$

### 2. LoRA Adapter Matrices:
- Adapter $A \in \mathbb{R}^{1 \times 2}$: $A = [0.5, \quad 1.0]$
- Adapter $B \in \mathbb{R}^{2 \times 1}$: $B = \begin{bmatrix} 2.0 \\ -1.0 \end{bmatrix}$

### Step 1: Base Model Transformation ($\vec{x} W_0$)
$$\vec{y}_{\text{base}} = [1.0, 2.0] \begin{bmatrix} 0.5 & -0.5 \\ 1.0 & 0.0 \end{bmatrix}$$
- Index 1: $(1.0 \times 0.5) + (2.0 \times 1.0) = 0.5 + 2.0 = 2.5$
- Index 2: $(1.0 \times -0.5) + (2.0 \times 0.0) = -0.5 + 0.0 = -0.5$
$$\vec{y}_{\text{base}} = [2.5, \quad -0.5]$$

### Step 2: LoRA Adapter Transformation ($(\vec{x} A^T) B^T$)
- First projection into bottleneck rank ($r=1$):
  $$\text{Intermediate} = \vec{x} \cdot A = (1.0 \times 0.5) + (2.0 \times 1.0) = 2.5$$
- Second projection up to dimension $k=2$:
  $$\Delta \vec{y}_{\text{raw}} = 2.5 \times B^T = 2.5 \times [2.0, \quad -1.0] = [5.0, \quad -2.5]$$
- Apply scaling factor ($\frac{\alpha}{r} = 2.0$):
  $$\Delta \vec{y} = 2.0 \times [5.0, \quad -2.5] = [10.0, \quad -5.0]$$

### Step 3: Combine Outputs
$$\vec{y}_{\text{final}} = \vec{y}_{\text{base}} + \Delta \vec{y} = [2.5, -0.5] + [10.0, -5.0] = [12.5, \quad -5.5]$$

Notice how the small $r=1$ matrices altered the output trajectory without touching or retraining any values in $W_0$.

---

## 3.9 Production Failure Modes & Engineering Audits

### 1. Catastrophic Forgetting
- **Symptom:** After fine-tuning a model on specialized legal or financial schemas, its score on general reasoning or standard coding benchmarks drops by 40%.
- **Root Cause:** The adaptation dataset lacked variety. Gradient updates optimized solely for specialized tokens, shifting weights away from general-domain minima.
- **Engineering Fix:** Implement **Experience Replay**. Blend 5–10% of general instruction-following data (e.g., SlimOrca or UltraChat) into the specialized training corpus.

### 2. Rank Overfitting & Inadequate Target Modules
- **Symptom:** Setting rank $r = 128$ yields worse validation metrics than $r = 16$.
- **Root Cause:** Setting $r$ too high introduces excess parameters that overfit to training noise. Conversely, targeting only $W_q$ and $W_v$ with high rank is less effective than targeting **all linear layers** ($W_q, W_k, W_v, W_o$, MLP projections) with a lower rank ($r=16$).
- **Audit Rule:** Always benchmark all-linear LoRA ($r=16, \alpha=32$) before increasing rank dimension.

### 3. Gradient Explosions in Half-Precision (`FP16`)
- **Symptom:** Training loss displays sudden `NaN` values at step 450 after hours of steady convergence.
- **Root Cause:** In standard `FP16`, the maximum representable value is $65,504$. During backpropagation, accumulated gradients exceeded this threshold, causing numeric overflow.
- **Engineering Fix:** Always fine-tune using **`BF16` (Brain Floating Point)**. While BF16 has lower precision than FP16 (7 bits vs 10 bits of mantissa), it preserves the wide 8-bit dynamic range of FP32 (up to $\approx 3.3 \times 10^{38}$), eliminating numeric overflow.

---

## 3.10 Chapter Summary Checkpoint

1. **Next-Token Prediction** uses Cross-Entropy Loss to measure predictive error, which backpropagation converts into weight gradients via the multivariate chain rule.
2. **Full Fine-Tuning** is constrained by optimizer memory: storing AdamW momentum, variance, and master weights requires **16–20 bytes per parameter**.
3. **LoRA** bypasses this memory wall by freezing $W_0$ and training low-rank factorized matrices ($B \times A$), cutting memory demand by over 80% without introducing extra inference latency after weights are merged.
4. **QLoRA** quantizes base weights into 4-bit NormalFloat (NF4) and uses Double Quantization to enable fine-tuning on consumer-grade hardware.
5. **DPO** achieves preference alignment mathematically without training a separate reward model or executing unstable PPO reinforcement learning loops.
