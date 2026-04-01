
from pydantic import BaseModel,Field
from typing import List, Optional, Dict, Any

class DebateAction(BaseModel):
    argument:str = Field(..., description="The textual argument submitted by the agent.")
    phase_tag: str = Field(..., description="The current phase of the debate (e.g., OPENING, REBUTTAL).")

    model_config = {
            "json_schema_extra": {
                "examples": [
                    {
                        "argument": "Regulating AGI globally is the only way to prevent an uncontrollable arms race between corporations.",
                        "phase_tag": "OPENING"
                    }
                ]
            }
        }
    
class DebateObservation(BaseModel):
    topic: str
    opponent_challenge:str
    done: bool = False
    reward: float = 0.0
    attempt_count: int = 0
    phase: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)

class DebateState(BaseModel):
    episode_id: Optional[str] = None
    step_count: int = 0
    current_phase: str = "opening"
    argument_count:int = 0
    history: List[Dict[str, Any]] = Field(default_factory=list)
    fallacies_detected: int = 0
    logical_score_sum: float = 0.0

