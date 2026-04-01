class SystemPrompt:
    def __init__(self, topic: str):
        self.topic = topic

    def get_prompt(self, phase: str, argument: str = "", opponent_challenge: str = "") -> str:
        """Dynamically generates a prompt based on the current debate phase."""
        
        # We use a dictionary to store templates with placeholders
        templates = {
            "OPENING": f"""
                Role: Debate Opponent. 
                Topic: {self.topic}. 
                Action: Acknowledge the opening claim: "{argument}" and set a skeptical stage.
            """,
            
            "CHALLENGE": f"""
                Role: Adversary. 
                Topic: {self.topic}. 
                Action: Attack the core claim: "{argument}". Identify logical gaps in this specific reasoning.
            """,
            
            "REBUTTAL": f"""
                Role: Critical Rebutter.
                The agent argued: "{argument}".
                Your previous challenge was: "{opponent_challenge}".
                Action: Point out why the agent's response fails to address your challenge.
            """,
            
            "CONSOLIDATION": f"""
                Role: Fact Checker.
                Topic: {self.topic}.
                Action: Question the evidence provided in: "{argument}". Focus on the weakest point.
            """,
            
            "CLOSING": f"""
                Role: Final Judge.
                Topic: {self.topic}.
                Action: Provide a final summary of the conversation. Evaluate if the agent remained consistent.
            """
        }
        
        # Return the specific phase prompt, or a default if not found
        return templates.get(phase.upper(), "Continue the debate logically.")