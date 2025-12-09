---
layout: default
title: "RL Sequence Modeling Blog"
---

# Transformers, take the Wheel! 
## Sequence Modeling in Offline RL


Reinforcement Learning (RL) has traditionally relied on value estimation and Bellman updates, which are often unstable and difficult to tune. 

This project explores a paradigm shift: treating RL as a Sequence Modeling problem. We analyze and replicate three Transformer-based approaches—Decision Transformer (DT), Trajectory Transformer (TT), and Iterative Energy Minimization (IEM)—to understand how language modeling architectures can solve decision-making tasks.


# What is reinforcement learning

<img src="RL.png" alt="Reinforcement Learning" style="max-width: 40%; height: auto; display: block; margin: 0 auto;">

Reinforcement learning is the third paradigm in machine learning after supervised and unsupervised learning. An agent wanders through an environment. At any moment it sits in some state $s$. It takes an action $a$. The world replies with a reward $r$ and shifts the agent to a new state $s'$.

Anytime an agent moves through a sequence of states, takes actions, and receives rewards, you’ve got a trajectory.You can think of it as the agent’s diary: every state visited, every choice made, and every pat-on-the-head (or slap-on-the-wrist) from the environment.   
$τ = (s₀, a₀, r₀, s₁, a₁, r₁, …, s_T) $


A policy is a probability distribution over actions:  
$π(a | s)$

The return—the total “goodness” of a trajectory—is the discounted sum of rewards:
$G_t = r_t + γ r_{t+1} + γ² r_{t+2} + … = Σ_{k=0}^{∞} γ^k r_{t+k}$  

The value function is just the expected return if you start in a state and follow the policy:
$V^π(s) = E_π [ G_t | s_t = s ]$  

The action-value function (Q-function) sharpens that by conditioning on the first action:
$Q^π(s, a) = E_π [ G_t | s_t = s, a_t = a ]$

The Bellman equations are where the recursion magic happens. They break down long-term value into “reward now plus value later”:
$V^π(s) = E_{a ~ π, s' ~ P} [ r(s, a) + γ V^π(s') ]$  
$Q*(s, a) = E_{s' ~ P} [ r(s, a) + γ max_{a'} Q*(s', a') ]$

For control (trying to find the best policy), you get the Bellman optimality equation:
$Q*(s, a) = E_{s' ~ P} [ r(s, a) + γ max_{a'} Q*(s', a') ]$



Reinforecment learning's become the backbone in many technologies. It's used in self driving cars, financial trades, recommendation systems, drones, robot manipulation etc.

# What is Offline Reinforcement learning

Offline reinforcement learning also known as batch reinforcement learning algorithms is the branch of RL that learns entirely from a fixed dataset of past interactions — no new exploration, no real-time environment access, just logged trajectories. 

What led to the creation of offline RL was that  many real-world systems generate mountains of logged data, yet letting an RL agent “explore” those systems would be unsafe, expensive, or outright impossible. Classic RL assumes the agent can poke the environment endlessly, but hospitals, factories, self-driving cars, financial markets, and even large-scale robotics labs don’t offer unlimited retries. You often have terabytes of past trajectories sitting around, but no permission to interact again. Researchers wanted a way to turn those static logs into policies without risking  exploration.

The challenge is that once the agent is trained, its policy may choose actions that never appeared in the dataset.   

# Transformers enter the chat: why sequence modeling for reinforcement learning



# Decision Transformer

 It establishes the baseline proof-of-concept models Reinforcement Learning as a Sequential modeling task.
Architecture used: causal GPT


# Trajectory Transformer


 IT accepts the premise of DT (RL is Sequence Modeling) but critiques the "blind" generation. To actively plan into the future, it adapts the NLP concept of Beam Search. 
Architecture used: causal GPT



# Iterative Energy Minimization
*Iterative Energy Minimization (IEM)*: "The Refiner" – Uses a BERT-like masked model to iteratively "denoise" and optimize a full plan at once, minimizing a learned energy function.

# Models and data sets

### HuggingFace Models Used

This project compares two transformer-based reinforcement learning models, both pretrained on the HalfCheetah dataset and available on HuggingFace:

### 1. Decision Transformer (DT)

**Model Identifier:** `edbeeching/decision-transformer-gym-halfcheetah-medium`

**Source:** HuggingFace Transformers Library  
**Model Type:** Decision Transformer  
**Architecture:** Transformer-based sequence model that conditions on return-to-go (R̂) to generate actions

**Key Characteristics:**
- Uses conditional generation based on desired return-to-go
- Input format: `[R̂, s, a, R̂, s, a, ...]` (interleaved return-to-go, states, and actions)
- Autoregressive action prediction
- Fast inference (single forward pass per action)

**Implementation Details:**
- Loaded via `DecisionTransformerModel.from_pretrained()` from `transformers` library
- Evaluated on `HalfCheetah-v4` environment from Gymnasium
- Default target return: 3600

---

### 2. Trajectory Transformer (TT)

**Model Identifier:** `CarlCochet/trajectory-transformer-halfcheetah-medium-v2`

**Source:** HuggingFace Transformers Library  
**Model Type:** Trajectory Transformer  
**Architecture:** Transformer-based model that learns joint distributions over full trajectories

**Key Characteristics:**
- Models complete trajectory distributions
- Input format: `[s, a, r, s, a, r, ...]` (interleaved states, actions, and rewards)
- Uses beam search for planning (explores multiple trajectory hypotheses)
- Better for long-horizon planning tasks

**Implementation Details:**
- Loaded via `TrajectoryTransformerModel.from_pretrained()` from `transformers` library
- Evaluated on `HalfCheetah-v4` environment from Gymnasium
- Supports beam search with configurable beam widths (K = 1, 2, 4, 8, 16, 32)

---

## HalfCheetah Dataset

### What is HalfCheetah?

**HalfCheetah** is a continuous control benchmark task from the MuJoCo physics simulator, part of the D4RL (Datasets for Deep Data-Driven Reinforcement Learning) benchmark suite. It is one of the most commonly used environments for evaluating offline reinforcement learning algorithms.

<div style="display: flex; justify-content: space-around; align-items: center; flex-wrap: wrap; gap: 20px; margin: 20px 0;">
  <img src="images/halfcheetah.png" alt="HalfCheetah Environment" style="max-width: 45%; height: auto;">
  <img src="images/half_cheetah.gif" alt="HalfCheetah Animation" style="max-width: 45%; height: auto;">
</div>

### Environment Description

**Task:** The agent controls a 2D cheetah robot (half of a full cheetah body) and must learn to run forward as fast as possible.
**State Space:**
- 17-dimensional continuous state vector
- Includes: body position, velocity, joint angles, joint velocities, and other kinematic features
**Action Space:**
- 6-dimensional continuous action space
- Represents torques applied to the 6 joints of the cheetah
**Reward Function:**
- Dense reward based on forward velocity
- Encourages the agent to run forward efficiently
- Typical episode returns range from ~0 to ~6000+ depending on policy quality

### Dataset Variants: "Medium" Quality

The "medium" suffix in the model names (`halfcheetah-medium`) refers to the **D4RL dataset quality level** used for pretraining:

**D4RL Dataset Quality Levels:**
- **random**: Trajectories from a random policy (lowest quality)
- **medium**: Trajectories from a partially trained policy (medium quality)
- **medium-replay**: Mix of medium-quality trajectories and some from replay buffer
- **medium-expert**: Mix of medium and expert-level trajectories
- **expert**: Trajectories from a fully trained expert policy (highest quality)

**"Medium" Dataset Characteristics:**
- Contains trajectories from a policy that achieves approximately 50-60% of expert performance
- Provides a good balance between diversity and quality
- Commonly used for offline RL research as it represents realistic scenarios where you have suboptimal but useful demonstration data
- Typically contains thousands of trajectories collected from the HalfCheetah-v4 environment

### Why HalfCheetah?

HalfCheetah is widely used in offline RL research because:
1. **Well-understood benchmark**: Established baseline with known performance metrics
2. **Continuous control**: Tests ability to handle continuous state and action spaces
3. **Dense rewards**: Provides learning signal throughout the episode
4. **Moderate complexity**: Not too simple (like CartPole) but not too complex (like humanoid), making it ideal for method development
5. **Standardized evaluation**: Part of D4RL, ensuring fair comparisons across papers


---

## HuggingFace Model Cards
  - DT: https://huggingface.co/edbeeching/decision-transformer-gym-halfcheetah-medium
  - TT: https://huggingface.co/CarlCochet/trajectory-transformer-halfcheetah-medium-v2


# Insights

## Inside the Black Box: Attention Analysis

<div style="display: flex; justify-content: space-around; align-items: center; flex-wrap: wrap;">
  <img src="images/dt_last_layer_attention map.png" alt="DT Last Layer Attention Map" style="max-width: 30%; height: auto;">
  <img src="images/tt_last_layer attention map.png" alt="TT Last Layer Attention Map" style="max-width: 30%; height: auto;">
  <img src="images/leap_baby_ai.png" alt="LEAP Baby AI" style="max-width: 30%; height: auto;">
</div>

DT: Vertical attention stripes confirm the model explicitly "checks" the desired future reward before committing to an action.  
TT: Strong diagonal banding reveals it focuses on immediate past context over long term past.  
IEM: Distributed grid-like attention states each position attends broadly across past AND future.  

# Novel Insights




# Limitations
Transformers are memory and computation expensive, using transformers in RL is unlikely given that deploying these in robots or real time environments woudl make them slow.

# Conclusion

The convergence of NLP and RL provides a unified framework where trajectories are treated as sentences, offering stability that traditional dynamic programming lacks. 
The future lies in hybrid architectures: combining sequence models' distributional robustness with Q-learning's trajectory stitching, and LEAP's iterative refinement for composable, adaptable planning.



# References
Chen, L., et al. (2021). Decision Transformer: Reinforcement Learning via Sequence Modeling. NeurIPS.   
Janner, M., et al. (2021). Offline Reinforcement Learning as One Big Sequence Modeling Problem. NeurIPS.  
Chen, H., et al. (2023). Planning with Sequence Models through Iterative Energy Minimization. ICLR.  

<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<script>
  window.MathJax = {
    tex: {
      inlineMath: [['$', '$'], ['\\(', '\\)']],
      displayMath: [['$$', '$$'], ['\\[', '\\]']]
    }
  };
</script>

