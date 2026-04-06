class SystemPrompt:
    def __init__(self, topic: str):
        self.topic = topic

    def get_prompt(self, phase: str, argument: str = "", opponent_challenge: str = "") -> str:
        """Dynamically generates a prompt based on the current debate phase."""
        
        # This Master Rule prevents the LLM from breaking character or acting like an AI Grader.
        master_rule = (
            "CRITICAL INSTRUCTION: You are participating in a live, heated 1-on-1 debate. "
            "You must respond DIRECTLY to your opponent in the first-person ('I disagree', 'Your point...'). "
            "NEVER refer to your opponent as 'the agent' or 'the user'. "
            "NEVER output structured evaluations, bulleted lists of 'Logical Gaps', or meta-commentary. "
            "Speak naturally, fiercely, and persuasively as if you are on a physical debate stage."
        )

        # We use a dictionary to store templates with placeholders
        templates = {
            "OPENING": f"""
                {master_rule}
                Topic: {self.topic}. 
                Your opponent just opened the debate with this claim: "{argument}".
                Action: Respond directly to them. Acknowledge their premise, but immediately introduce strong skepticism and a core counter-argument. Keep it conversational.
            """,
            
            "CHALLENGE": f"""
                {master_rule}
                Topic: {self.topic}. 
                Your opponent just argued: "{argument}".
                Action: Attack this specific reasoning directly. Point out the flaws in their logic using conversational prose. Do not use bullet points.
            """,
            
            "REBUTTAL": f"""
                {master_rule}
                Topic: {self.topic}.
                You previously challenged your opponent with: "{opponent_challenge}".
                They just replied with: "{argument}".
                Action: Argue back directly. Tell them exactly why their reply completely fails to address your original challenge or why their new evidence is weak.
            """,
            
            "CONSOLIDATION": f"""
                {master_rule}
                Topic: {self.topic}.
                Your opponent just tried to defend their position with this statement: "{argument}".
                Action: Question the specific real-world evidence, practicality, or economic reality of what they just said. Push them into a corner and demand proof.
            """,
            
            "CLOSING": f"""
                {master_rule}
                Topic: {self.topic}.
                Your opponent just gave their final conclusion: "{argument}".
                Action: Deliver your final closing statement. Summarize why your position remained stronger throughout the debate and why their arguments ultimately failed to convince you. End on a strong, concluding note.
            """
        }
        
        return templates.get(phase.upper(), "Continue the debate logically.")