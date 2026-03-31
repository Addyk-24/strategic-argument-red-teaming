from dotenv import load_dotenv
load_dotenv()

import os
from groq import Groq

from schema.schemas import DebateState,DebateObservation,DebateAction
from reward_metrics.reward_metrics import RewardMetrics
from uuid import uuid4
from collections import Counter


import logging

from openai import OpenAI

from prompter.system_prompt import SystemPrompt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


api_key = os.getenv("HF_TOKEN")


class DebateEnvironment:

    def __init__(self):
        self._state = DebateState()
        self.picked_topic = ""
        self.system_prompt = SystemPrompt(topic=self.picked_topic)
        self.reward_metrics = RewardMetrics()

    def reset(self,topic:str) -> DebateObservation:

        attempt_count = 0


        self.picked_topic = topic
        self._state = DebateState(
            episode_id=str(uuid4()),
            step_count=0,
            current_phase="opening",
        )

        return DebateObservation(
            topic=self.picked_topic,
            opponent_challenge="",
            attempt_count=attempt_count,
            phase=self._state.current_phase
        )
    
    def state(self) -> DebateState:
        return self._state
    
    # claim → challenge → rebuttal - 3 steps - prev arch
    # opening → argument (steps 2-4) → closing - updated arch
    
    def step(self,action:DebateAction ) -> DebateObservation:

        phases = ["OPENING", "CHALLENGE", "REBUTTAL", "CONSOLIDATION", "CLOSING"]

        if self._state.step_count < 5:
                    self._state.current_phase = phases[self._state.step_count]

        opponent_response = self._get_opponent_response(action)

        reward = self._calculate_reward(action)


        self._state.step_count += 1
        is_done = self._state.step_count >= 5


        return DebateObservation(
            topic=self.picked_topic,
            opponent_challenge= opponent_response if not is_done else "" ,
            done=is_done,
            reward=reward,
            attempt_count=self._state.step_count,
            phase=self._state.current_phase,
            metadata=self._state.history[-1] if self._state.history else {}
        )

                
    def _calculate_reward(self,action:DebateAction) -> float:

        reward = 0.0
        text = action.argument.lower()
        agent_history = [h["action"] for h in self._state.history if h["role"] == "agent"]
        opponent_history = [h["action"] for h in self._state.history if h["role"] == "opponent"]

        coverage = self.reward_metrics.argument_coverage(agent_history,text)
        opp_refutation = self.reward_metrics.opponent_coverage(opponent_history,text)
        synthesis = self.reward_metrics.synthesis_score(text)


        if action.phase_tag != self._state.current_phase:
            reward -= 1.0
        
        
        if len(action.argument.split()) < 10:
            reward -= 0.5
        else:
            reward += 0.2

        if self._state.current_phase == "OPENING":
            if action.phase_tag == "OPENING":
                reward += self.reward_metrics.impact_score(text) * 0.1

        elif self._state.current_phase in ["CHALLENGE", "REBUTTAL"]:
            ref_score = self.reward_metrics.cal_refu_score(opponent_history,text)

            reward += ref_score * 0.2
        elif self._state.current_phase == "CLOSING":
            reward += (coverage * 0.5) + (opp_refutation * 0.3) + (synthesis * 0.2)
        
        if len(agent_history) >= 2:
            if self.reward_metrics.similarity(text,agent_history[-2]) > 0.8:
                reward -= 1.0

        words = text.split()

        if words:

            freq = Counter(words)

            total = len(words)
            
            for word,count in freq.items():
                if count/total >= 0.9:
                    reward -= 2.0

        return max(0.0, min(1.0, reward))
    

            
    def _get_opponent_response(self, action: DebateAction) -> str:
            phase = self._state.current_phase.upper()
            

            prev_opponent_challenge = ""
            for entry in reversed(self._state.history):
                if entry.get("role") == "opponent":
                    prev_opponent_challenge = entry.get("action", "")
                    break
                    
            current_prompt = self.system_prompt.get_prompt(
                phase=phase,
                argument=action.argument,
                opponent_challenge=prev_opponent_challenge
            )
            
            self._state.history.append({
                "role": "agent",
                "phase": phase,
                "action": action.argument,
                "prompt": current_prompt 
            })

            if phase in ("OPENING", "CHALLENGE", "REBUTTAL", "CONSOLIDATION", "CLOSING"):
                
                response = self.inference(current_prompt)
                
                self._state.history.append({
                    "role": "opponent",
                    "phase": phase,
                    "action": response,
                    "prompt": current_prompt 
                })
                
                return response
                
            else:
                logger.info(f"Invalid Environment Phase: {phase}")
                return "Error: Invalid phase."


    def inference(self, prompt: str) -> str:

            client = Groq(
                api_key=api_key,
            )
            
            try:
                response = client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],

                    model="llama-3.1-8b-instant", 
                    temperature=0.7,
                    max_tokens=500,
                )
                
                return response.choices[0].message.content
                
            except Exception as e:
                print(f"Groq API Error: {e}")
                return "Error: Could not generate response."
    
