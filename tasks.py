from schema.schemas import DebateObservation
from reward_metrics.reward_metrics import RewardMetrics


metrics = RewardMetrics()

class Task1_SingleClaim:
    """
    EASY: Can the agent formulate a strong, logical opening statement?
    Evaluated after step 1.
    """
    name = "single_claim"
    difficulty = "easy"
    
    def grade(self, observation: DebateObservation) -> float:
        score = 0.0
        
        action_text = observation.metadata.get("action", "")
        
        # Check 1: Length threshold (0.4 points)
        if len(action_text.split()) >= 10:
            score += 0.4
            
        # Check 2: Reasoning Keyword Presence (0.4 points)
        reasoning_keywords = ["because", "therefore", "however", "consequently", "shows", "proves"]
        if any(kw in action_text.lower() for kw in reasoning_keywords):
            score += 0.4
            
        # Check 3: Phase validation (0.2 points)
        if observation.metadata.get("phase") == "OPENING":
            score += 0.2
            
        return min(1.0, score)


class Task2_ClaimAndRebuttal:
    """
    MEDIUM: Can the agent survive a back-and-forth and deliver a strong rebuttal?
    Evaluated after step 3 (Opening -> Challenge -> Rebuttal).
    """
    name = "claim_and_rebuttal"  
    difficulty = "medium"
    
    def grade(self, observation: DebateObservation) -> float:
        # Fail immediately if they didn't reach the required depth (crashed or quit early)
        if observation.attempt_count < 3:
            return 0.0  
            
        score = 0.0
        action_text = observation.metadata.get("action", "")
        
        # 1. Base points for reaching the correct phase successfully
        score += 0.3 
        
        # 2. Direct Evaluation: Is it actually a rebuttal? 

        refutation_strength = metrics.refutation_strength(action_text)
        score += (refutation_strength * 0.4) 
            
        if len(action_text.split()) > 15:
            score += 0.3
            
        return min(1.0, score)


class Task3_FullDebate:
    """
    HARD: Can the agent complete a 5-turn debate and successfully synthesize 
    the arguments into a concluding statement?
    Evaluated at the end of the episode (step 5).
    """
    name = "full_debate"
    difficulty = "hard"
    
    def grade(self, observation: DebateObservation) -> float:
        if not observation.done or observation.attempt_count < 5:
            return 0.0
            
        score = 0.0
        action_text = observation.metadata.get("action", "")
        
        # 1. Base points for surviving all 5 turns
        score += 0.2
        
        # 2. Direct Evaluation: Synthesis

        synthesis = metrics.synthesis_score(action_text)
        score += (synthesis * 0.4)
        
        # 3. Direct Evaluation: Impact

        impact = metrics.impact_score(action_text)
        score += (min(impact, 2) * 0.1)
        
        # 4. Length check: A closing summary must be substantial
        if len(action_text.split()) >= 20:
            score += 0.3
            
        return max(0.0, min(1.0, score))