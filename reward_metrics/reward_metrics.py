#similarity runs a full encode on every call — if called multiple times per step it's slow. This matters for training loops. Consider caching or batching, but not urgent right now.

from sentence_transformers import SentenceTransformer, util
from transformers import logging as transformers_logging
import logging
import torch

transformers_logging.set_verbosity_error()

model = SentenceTransformer('all-MiniLM-L6-v2')


class RewardMetrics:
    def __init__(self):
        self.LINKING_PHRASES = [
        "taken together", "overall", "this shows", 
        "together", "therefore", "thus", "in conclusion"
    ]
        self.NEGATION_PATTERNS = [
        "fails", "incorrect", "wrong", "does not", 
        "untrue", "contradiction", "weak"
    ]
        self.IMPACT_PHRASES = [
        "we win", "therefore", "in conclusion",
        "the key takeaway", "this proves"
    ]
        self._embedding_cache = {}
    
    def _get_embedding(self,text:str):
        """Fetches embedding from cache, or computes it if not found."""
        if text not in self._embedding_cache:
            self._embedding_cache[text] = model.encode(text,convert_to_tensor=True)
        
        return self._embedding_cache[text]

    def similarity(self,a,b):
        emb1 = self._get_embedding(a)
        emb2 = self._get_embedding(b)
        return util.cos_sim(emb1, emb2).item()


    def argument_coverage(self,agent_args, closing, threshold=0.5):

        if not agent_args:
            return 0.0
        
        closing_emb = self._get_embedding(closing)
        history_embs = [self._get_embedding(arg) for arg in agent_args]



        # Stacking them into a single 2D PyTorch tensor
        history_matrix = torch.stack(history_embs)

        similarities = util.cos_sim(closing_emb, history_matrix)

        # similarities is a tensor of shape [1, num_args]
        covered = (similarities > threshold).sum().item()

        return covered / len(agent_args)


    def synthesis_score(self,text):
        count = sum(1 for phrase in self.LINKING_PHRASES if phrase in text.lower())
        return min(count / len(self.LINKING_PHRASES), 1.0)

    def opponent_coverage(self,opponent_args, closing, threshold=0.5):

        if not opponent_args:
            return 0.0
        
        closing_emb = self._get_embedding(closing)
        history_embs = [self._get_embedding(arg) for arg in opponent_args]
        


        # Stacking them into a single 2D PyTorch tensor
        history_matrix = torch.stack(history_embs)

        similarities = util.cos_sim(closing_emb, history_matrix)

        # similarities is a tensor of shape [1, num_args]
        covered = (similarities > threshold).sum().item()

        return covered / len(opponent_args)
    

    def refutation_strength(self,text):
        count = sum(1 for word in self.NEGATION_PATTERNS if word in text.lower())
        return min(count / len(self.NEGATION_PATTERNS), 1.0)

    def cal_refu_score(self,opponent_args,text:str):
        coverage = self.opponent_coverage(opponent_args,text)
        strength = self.refutation_strength(text)
        
        refutation_score = (coverage * 0.7) + (strength* 0.3) 
        
        return refutation_score
    

    def impact_score(self,text):
        count = 0
        for phrase in self.IMPACT_PHRASES:
            if phrase in text.lower():
                count += 1
        
        # bonus if appears near end
        if any(phrase in text.lower()[-100:] for phrase in self.IMPACT_PHRASES):
            count += 1
            
        return min(count / (len(self.IMPACT_PHRASES) + 1), 1.0)