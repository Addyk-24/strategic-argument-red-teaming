"""
grpo_train.py
Simple GRPO-style training loop for the Debate Coach Environment.
Tracks reward improvement across episodes to demonstrate learning signal.
"""

from dotenv import load_dotenv
load_dotenv()

import os
import time
import json
from collections import defaultdict
from openai import OpenAI

from envs.environment import DebateEnvironment
from models.schemas import DebateAction
from graders.tasks import Task1_SingleClaim, Task2_ClaimAndRebuttal, Task3_FullDebate

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL")
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")
BENCHMARK = "strategic-argument-red-teaming"

NUM_EPISODES       = 30   # total training episodes
GROUP_SIZE         = 5    # GRPO: episodes per group for baseline calculation
SLEEP_BETWEEN_CALLS = 1.0 # seconds — Groq rate limit protection

TOPICS = [
    "Universal Basic Income is necessary for the future economy.",
    "Artificial Intelligence should be strictly regulated by governments.",
    "Social media does more harm than good to society.",
    "Climate change requires immediate radical policy action.",
    "Remote work is more productive than office work.",
]

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN,timeout=1800,max_retries=2,)

def generate_argument(topic: str, phase: str, opponent_challenge: str,
                      temperature: float = 0.7) -> str:
    """
    The 'policy' — generates an argument given the current observation.
    Temperature is increased over time to encourage exploration (GRPO-style).
    """
    prompt = f"You are a skilled debater. Topic: '{topic}'.\n"
    prompt += f"Current debate phase: {phase}.\n"

    if opponent_challenge:
        prompt += f"Opponent argued: '{opponent_challenge}'\n"
        prompt += ("Respond directly and logically. "
                   "Use words like 'therefore', 'because', 'however', "
                   "'this fails because', 'the evidence shows'.\n")
    else:
        prompt += ("Make a strong opening claim. "
                   "Use 'therefore' or 'because' to show reasoning.\n")

    prompt += "Be concise: 30-60 words. No filler phrases."

    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=200,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"  [Agent API Error] {e}")
        return "therefore this position is correct because the evidence supports it."


def run_episode(env: DebateEnvironment, topic: str,
                grader, temperature: float = 0.7) -> dict:
    """
    Runs one full episode and returns a result dict with:
      - per-step rewards from the environment
      - final task score from the grader
    """
    obs = env.reset(topic)
    step_rewards = []

    while not obs.done:
        argument = generate_argument(
            topic=obs.topic,
            phase=obs.phase.upper(),
            opponent_challenge=obs.opponent_challenge,
            temperature=temperature,
        )
        action = DebateAction(argument=argument, phase_tag=obs.phase.upper())
        obs = env.step(action)
        step_rewards.append(obs.reward)
        time.sleep(SLEEP_BETWEEN_CALLS)

    final_score = grader.grade(obs)

    return {
        "topic":        topic,
        "step_rewards": step_rewards,
        "mean_reward":  sum(step_rewards) / len(step_rewards) if step_rewards else 0.0,
        "final_score":  final_score,
    }


# GRPO core
def compute_grpo_advantage(group_results: list[dict]) -> list[float]:
    """
    GRPO: advantage = (score - group_mean) / (group_std + epsilon)
    This tells us which episodes in the group were above/below average.
    A positive advantage = this episode's policy was better than the group baseline.
    """
    scores = [r["final_score"] for r in group_results]
    mean   = sum(scores) / len(scores)
    variance = sum((s - mean) ** 2 for s in scores) / len(scores)
    std    = variance ** 0.5

    advantages = [(s - mean) / (std + 1e-8) for s in scores]
    return advantages


# Training loop
def train():
    env    = DebateEnvironment()
    grader = Task3_FullDebate()

    all_results   = []
    group_buffer  = []
    episode_scores = []

    # Temperature annealing: start high (explore) → end lower (exploit)
    def get_temperature(episode: int) -> float:
        return max(0.4, 1.0 - (episode / NUM_EPISODES) * 0.6)

    print("=" * 50)
    print("GRPO Training — Debate Coach Environment")
    print(f"Episodes: {NUM_EPISODES} | Group size: {GROUP_SIZE}")
    print(f"Model: {MODEL_NAME}")
    print("=" * 50)

    for episode in range(NUM_EPISODES):
        topic = TOPICS[episode % len(TOPICS)]
        temp  = get_temperature(episode)

        print(f"\nEpisode {episode + 1}/{NUM_EPISODES} | "
              f"Topic: {topic[:40]}... | Temp: {temp:.2f}")

        result = run_episode(env, topic, grader, temperature=temp)
        all_results.append(result)
        group_buffer.append(result)
        episode_scores.append(result["final_score"])

        print(f"  Step rewards: {[f'{r:.2f}' for r in result['step_rewards']]}")
        print(f"  Mean reward:  {result['mean_reward']:.3f}")
        print(f"  Final score:  {result['final_score']:.3f}")

        # Every GROUP_SIZE episodes: compute GRPO advantage for the group
        if len(group_buffer) == GROUP_SIZE:
            advantages = compute_grpo_advantage(group_buffer)
            group_scores = [r["final_score"] for r in group_buffer]

            print(f"\n  --- Group {episode // GROUP_SIZE + 1} Summary ---")
            print(f"  Scores:     {[f'{s:.2f}' for s in group_scores]}")
            print(f"  Advantages: {[f'{a:.2f}' for a in advantages]}")
            print(f"  Best episode in group: "
                  f"Episode {episode - GROUP_SIZE + 1 + group_scores.index(max(group_scores)) + 1} "
                  f"(score={max(group_scores):.2f})")

            # In real GRPO: you would use advantages to weight your policy gradient update.
            # Here we log them so you can see the signal your environment produces.
            for i, (res, adv) in enumerate(zip(group_buffer, advantages)):
                res["advantage"] = adv

            group_buffer = []

    # Final report
    print("\n" + "=" * 50)
    print("TRAINING COMPLETE — Reward Curve")
    print("=" * 50)

    # Print reward curve in 5-episode windows
    window = 5
    for i in range(0, NUM_EPISODES, window):
        chunk  = episode_scores[i:i + window]
        avg    = sum(chunk) / len(chunk)
        bar    = "█" * int(avg * 20)
        print(f"  Episodes {i+1:2d}-{i+len(chunk):2d}: {avg:.3f}  {bar}")

    overall_avg = sum(episode_scores) / len(episode_scores)
    first_half  = sum(episode_scores[:NUM_EPISODES//2]) / (NUM_EPISODES // 2)
    second_half = sum(episode_scores[NUM_EPISODES//2:]) / (NUM_EPISODES - NUM_EPISODES // 2)

    print(f"\n  Overall average:     {overall_avg:.3f}")
    print(f"  First half average:  {first_half:.3f}")
    print(f"  Second half average: {second_half:.3f}")
    print(f"  Trend: {'IMPROVING ↑' if second_half > first_half else 'DECLINING ↓'}")

    # Save results to JSON for your README / submission
    with open("training_results.json", "w") as f:
        json.dump({
            "model":          MODEL_NAME,
            "num_episodes":   NUM_EPISODES,
            "episode_scores": episode_scores,
            "overall_avg":    overall_avg,
            "first_half_avg": first_half,
            "second_half_avg": second_half,
            "all_results":    all_results,
        }, f, indent=2)

    print("\n  Results saved to training_results.json")
    print("=" * 50)


if __name__ == "__main__":
    train()