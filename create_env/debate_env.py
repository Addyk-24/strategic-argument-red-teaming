from schema.schemas import DebateState,DebateObservation
from uuid import uuid4

class DebateEnvironment:

    def __init__(self):
        self._state = DebateState()
        self.picked_topic = ""

    def reset(self,topic:str,opponent_challenge:str) -> DebateObservation:

        attempt_count = 0

        self.picked_topic = topic
        self._state = DebateState(
            episode_id=str(uuid4()),
            step_count=0,
            current_phase="opening",
        )

        return DebateObservation(
            topic=self.picked_topic,
            opponent_challenge=opponent_challenge,
            attempt_count=attempt_count,
            phase=self._state.current_phase
        )
    



