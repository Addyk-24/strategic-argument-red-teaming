from dotenv import load_dotenv
load_dotenv()

"""
Inference Script
===================================
Compliant with Hackathon Mandatory Variables.
"""

import os
import time
from openai import OpenAI


from environment import DebateEnvironment
from schema.schemas import DebateAction
from tasks import Task1_SingleClaim, Task2_ClaimAndRebuttal, Task3_FullDebate

API_BASE_URL = os.getenv("API_BASE_URL")
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_NAME = os.getenv("MODEL_NAME")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)


def generate_agent_argument(topic: str, phase: str, opponent_challenge: str) -> str:
    """Uses the injected LLM to generate the agent's move based on the observation."""
    prompt = f"You are a skilled debater. The topic is: '{topic}'.\n"
    prompt += f"The current phase of the debate is: {phase}.\n"
    
    if opponent_challenge:
        prompt += f"Your opponent just argued: '{opponent_challenge}'\n"
        prompt += "Write a direct, logical response to their challenge. Use reasoning keywords like 'therefore' or 'because'.\n"
    else:
        prompt += "Write a strong, logical opening statement for your side. Use reasoning keywords like 'therefore' or 'because'.\n"
        
    prompt += "Keep your response under 50 words and do not include any conversational filler."

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=150
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Agent API Error: {e}")
        return "therefore I agree."

def evaluate_baseline():
    env = DebateEnvironment()
    topic = "Universal Basic Income is necessary for the future economy."
    
    print(f"Starting Baseline Evaluation with {MODEL_NAME}")
    print(f"Topic: {topic}\n")
    print("-" * 40)


    print("Evaluating Task 1: Single Claim (1 Step)")
    obs = env.reset(topic)
    
    argument = generate_agent_argument(obs.topic, obs.phase, obs.opponent_challenge)
    action = DebateAction(argument=argument, phase_tag=obs.phase.upper())
    obs = env.step(action)
    
    task1 = Task1_SingleClaim()
    score1 = task1.grade(obs)
    print(f"Task 1 Score: {score1:.2f}\n")
    time.sleep(1)


    print("-" * 40)
    print("Evaluating Task 2: Claim and Rebuttal (3 Steps)")
    obs = env.reset(topic)
    
    for _ in range(3):
        argument = generate_agent_argument(obs.topic, obs.phase, obs.opponent_challenge)
        action = DebateAction(argument=argument, phase_tag=obs.phase.upper())
        obs = env.step(action)
        time.sleep(1)
        
    task2 = Task2_ClaimAndRebuttal()
    score2 = task2.grade(obs)
    print(f"Task 2 Score: {score2:.2f}\n")


    print("-" * 40)
    print("Evaluating Task 3: Full Debate (5 Steps)")
    obs = env.reset(topic)
    
    while not obs.done:
        argument = generate_agent_argument(obs.topic, obs.phase, obs.opponent_challenge)
        action = DebateAction(argument=argument, phase_tag=obs.phase.upper())
        obs = env.step(action)
        time.sleep(1)
        
    task3 = Task3_FullDebate()
    score3 = task3.grade(obs)
    print(f"Task 3 Score: {score3:.2f}\n")

    print("=" * 40)
    print("BASELINE EVALUATION COMPLETE")
    print(f"Task 1 (Easy):   {score1:.2f} / 1.00")
    print(f"Task 2 (Medium): {score2:.2f} / 1.00")
    print(f"Task 3 (Hard):   {score3:.2f} / 1.00")
    print("=" * 40)

if __name__ == "__main__":
    evaluate_baseline()