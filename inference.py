import os
import time
from typing import List, Optional
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

from envs.environment import DebateEnvironment
from models.schemas import DebateAction
import logging

logging.getLogger("httpx").setLevel(logging.WARNING)

API_BASE_URL = os.getenv("API_BASE_URL")
api_key = os.getenv("HF_TOKEN")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")
BENCHMARK = "strategic-argument-red-teaming"

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=api_key,
    timeout=1800,
    max_retries=2,
)

# STDOUT LOGGING FUNCTIONS 
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error.replace('\n', ' ') if error else "null"
    done_val = str(done).lower()
    # Action string must not contain newlines to avoid breaking the parser
    safe_action = action.replace('\n', ' ') 
    print(
        f"[STEP] step={step} action={safe_action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

# Core Logic
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
        return f"ERROR: {e}"

# EVALUATION LOOP
# --- EVALUATION LOOP ---

def evaluate_task(env, topic: str, task_name: str, max_steps: int):
    """Runs a single task and emits strict logs."""
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)
    
    obs = env.reset(topic)
    step_count = 0
    rewards = []
    error = None
    success = False
    
    try:
        while step_count < max_steps and not obs.done:
            step_count += 1
            
            argument = generate_agent_argument(obs.topic, obs.phase, obs.opponent_challenge)
            
            if argument.startswith("ERROR:"):
                error = argument
                action = DebateAction(argument="Pass.", phase_tag=obs.phase.upper())
            else:
                action = DebateAction(argument=argument, phase_tag=obs.phase.upper())
            
            obs = env.step(action)
            reward = obs.reward if obs.reward is not None else 0.0
            rewards.append(reward)
            
            log_step(step=step_count, action=argument, reward=reward, done=obs.done, error=error)
            time.sleep(1) 
            
        # Success is defined as getting a positive score across the task
        total_score = sum(rewards)
        success = total_score > 0.0 
        
    except Exception as e:
        error = str(e)
        success = False
    finally:
        log_end(success=success, steps=step_count, score=sum(rewards), rewards=rewards)


def evaluate_baseline():
    
    env = DebateEnvironment()
    topic = "Universal Basic Income is necessary for the future economy."
    

    evaluate_task(env, topic, task_name="Task1_SingleClaim", max_steps=1)
    

    evaluate_task(env, topic, task_name="Task2_ClaimAndRebuttal", max_steps=3)
    
    evaluate_task(env, topic, task_name="Task3_FullDebate", max_steps=5)
    
    try:
        env.close()
    except:
        pass

if __name__ == "__main__":
    evaluate_baseline()

    