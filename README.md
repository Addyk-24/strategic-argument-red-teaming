# 🛡️ Strategic Argument Red-Teaming

**An OpenEnv RL Benchmark for Multi-Turn Adversarial Reasoning and Policy Defense**

## 🧠 The Problem: The "Multi-Turn" Vulnerability in LLMs

Modern Large Language Models (LLMs) are heavily tested against single-shot prompt injections (e.g., "ignore previous instructions"). However, real-world manipulation and misalignment rarely happen in a single turn.

Evaluating an LLM's robustness against sustained, logical persuasion over a long conversational horizon is incredibly difficult. Currently, testing this requires either expensive human red-teaming or static benchmarks that fail to capture the dynamic, evolving nature of a real debate. There is a lack of automated, reproducible environments to train and evaluate AI agents on long-term strategic argumentation.

## 💡 The Solution: Strategic Argument Red Teaming

This project introduces a custom Reinforcement Learning (RL) environment built on `openenv-core` designed to simulate and evaluate multi-turn adversarial debates.

Instead of a simple question-and-answer format, this environment forces an Attacker Agent to systematically dismantle a Defender Model's arguments over a long-running trajectory.

**Key Innovations:**
- **Objective Evaluation:** The environment features a custom reward function that dynamically grades the agent based on argument coverage, logical refutation, and synthesis, while heavily penalizing repetitive dialogue.
- **Scalable Trajectories:** The environment is structured into escalating evaluation tasks (ranging from 1-step claims to 5+ step full debates), allowing researchers to test both short-horizon tactics and delayed-gratification strategies.
- **Provider Agnostic:** Fully integrated with the OpenAI client standard, allowing researchers to plug and play different open-source models via Hugging Face or Groq to test different model alignments against each other.

---

## ⚙️ Environment Mechanics

### The State Machine (5 Phases)
The environment enforces strict episode boundaries through a 5-turn state machine. The agent must adapt its strategy based on the current phase:
1. **OPENING:** Formulate a strong, initial logical claim.
2. **CHALLENGE:** The opponent attacks the core claim to expose logical gaps.
3. **REBUTTAL:** The agent must directly and forcefully address the opponent's challenge.
4. **CONSOLIDATION:** The opponent questions the underlying evidence of the rebuttal.
5. **CLOSING:** The agent must semantically synthesize the entire conversation into a concluding statement.

### Action & Observation Spaces
* **Action (`DebateAction`)**: 
  * `argument` (str): The raw text of the agent's move.
  * `phase_tag` (str): The agent's awareness of the current environment phase.
* **Observation (`DebateObservation`)**: 
  * `topic` (str): The debate topic.
  * `opponent_challenge` (str): The dynamically generated counter-argument from the AI adversary.
  * `phase` (str): The current phase of the debate.
  * `reward` (float): The step-by-step reward signal `[-1.0, 1.0]`.
  * `done` (bool): Episode termination flag.

---

## 🧠 Meaningful Reward Function & Semantic Shaping
To provide dense, partial progress signals for GRPO training, the reward function uses **SentenceTransformers (`all-MiniLM-L6-v2`)** rather than simple keyword matching. 

* **Semantic Coverage Scoring:** In the `CLOSING` phase, the environment computes vectorized Cosine Similarities between the agent's final statement and the history of both the agent's and opponent's previous arguments. High overlap yields high rewards, teaching the model to actively *synthesize* rather than ignore the opponent.
* **Anti-Reward Hacking (Repetition Penalty):** The environment tracks the agent's embedded history. If the agent repeats its own previous argument (Cosine Similarity > 0.8), it receives a harsh `-1.0` penalty, forcing novel generation.
* **Bounded Vector Caching:** To ensure the environment runs blazingly fast during RL training loops, embeddings are cached in a FIFO bounded dictionary, preventing memory leaks (OOM) over thousands of episodes.

---

## 🎯 Evaluation Tasks & Graders
The environment includes three deterministic graders. **Crucially, the graders evaluate the raw text of the agent's metadata, completely decoupled from the environment's internal training reward.** This prevents the agent from simply "gaming the training math" during evaluation.

| Task | Difficulty | Objective & Grading Criteria |
| :--- | :--- | :--- |
| **Task 1: Single Claim** | **Easy** | Evaluates the formulation of the opening statement. Graded heavily on length thresholds and the presence of logical structuring keywords (e.g., "therefore", "consequently"). |
| **Task 2: Claim & Rebuttal** | **Medium** | Evaluates 3-turn survival. Graded on the textual strength of the rebuttal (use of refutation patterns) and whether the agent successfully generated novel text without triggering the environment's repetition penalty. |
| **Task 3: Full Debate Synthesis** | **Hard** | Evaluates full 5-turn survival. Graded strictly on the inclusion of semantic synthesis phrases and impactful concluding statements. |

---

## 📊 Baseline Inference Scores
A baseline script (`inference.py`) is provided. It uses the mandatory **OpenAI API Client** (routed to Hugging Face's serverless infrastructure using `Qwen2.5-72B-Instruct`) to prove the environment provides a perfect learning gradient with substantial headroom for RL fine-tuning.

* **Task 1 (Easy):** `1.00 / 1.00` (The base model easily formulates initial claims).
* **Task 2 (Medium):** `0.71 / 1.00` (The model survives but struggles to deliver mathematically perfect refutations).
* **Task 3 (Hard):** `0.50 / 1.00` (The model survives 5 turns but frequently fails to fully synthesize the opponent's arguments, proving significant headroom for GRPO training).

---

## 🚀 Setup & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Addyk-24/Debate-Coach-Environment.git
cd Debate-Coach-Environment
```
### 2. Set Up Environment Variables
   ```bash
   API_BASE_URL="openai_client_hf_url_here" 
   HF_TOKEN="hf__key_here"
   MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
   ```
### 3. Install dependencies:
   ```bash
    pip install -r requirements.txt
   ```
### 4. Run the Environment Server:
   ```bash
    python inference.py
   ```

## 👨‍💻 Author
Aditya Katkar
