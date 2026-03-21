
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

@dataclass
class DebateAction:
    argument:str
    phase_tag: str
    
@dataclass
class DebateObservation:
    topic: str
    opponent_challenge:str
    attempt_count: int
    phase: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DebateState:
    episode_id: Optional[str] = None
    step_count: int = 0
    current_phase: str
