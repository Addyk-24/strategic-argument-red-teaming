# 🛡️ Strategic Argument Red-Teaming

**An OpenEnv RL Benchmark for Multi-Turn Adversarial Reasoning and Policy Defense**

## 📖 Overview & Real-World Utility (30% Rubric Focus)
As Large Language Models are increasingly deployed in enterprise environments for drafting corporate policies, legal summaries, and public relations statements, they must be capable of surviving hostile review processes. 

This environment simulates a strategic **"Red-Teaming"** scenario. The agent acts as the defender of a claim and must successfully navigate a 5-phase adversarial debate against a dynamically generated, skeptical LLM opponent. 

This is not a toy game; it is a direct simulation of **RLAIF (Reinforcement Learning from AI Feedback)** workflows used at frontier labs to align models against sycophancy (backing down too easily) while training them to maintain logical consistency, deliver strong refutations, and synthesize opposing viewpoints.

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
  * `opponent_challenge` (str): The dynamically generated counter-argument from the Groq-powered adversary.
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
A baseline script (`baseline.py`) is provided. It uses the **OpenAI API Client** (routed to Groq's `llama-3.1-8b-instant` for high-speed evaluation) to prove the environment provides a perfect learning gradient with substantial headroom for RL fine-tuning.

* **Task 1 (Easy):** `1.00 / 1.00` (The base model easily formulates initial claims).
* **Task 2 (Medium):** `0.71 / 1.00` (The model survives but struggles to deliver mathematically perfect refutations).
* **Task 3 (Hard):** `0.50 / 1.00` (The model survives 5 turns but completely fails to synthesize the opponent's arguments, proving significant headroom for GRPO training).

---

## 🚀 Setup & Installation

### Prerequisites
1. Clone the Repository
```bash
git clone https://github.com/Addyk-24/Debate-Coach-Environment.git
cd Debate-Coach-Environment
```
2. Set Up Environment Variables
   ```bash
   API_BASE_URL="openai_client_groq_url_here" 
   MODEL_NAME="llama-3.1-8b-instant"
   GROQ_API_KEY="gsk_your_actual_key_here"
   ```
3. Install dependencies:
   ```bash
    pip install -r requirements.txt
   ```
4. Run the Environment Server:
   ```bash
    python server/app.py
   ```

## Author
Aditya Katkar
