from sacrebleu import BLEU
from typing import List


def calculate_bleu(references: List[str], hypotheses: List[str]) -> float:
    """
    Calculate BLEU score between references and hypotheses.
    
    Args:
        references: List of reference translations
        hypotheses: List of predicted translations
    
    Returns:
        BLEU score
    """
    if len(references) != len(hypotheses):
        raise ValueError("References and hypotheses must have the same length")
    
    # Convert to format expected by sacrebleu
    refs = [[ref.split()] for ref in references]
    hyps = [hyp.split() for hyp in hypotheses]
    
    bleu = BLEU()
    score = bleu.corpus_score(hyps, refs)
    
    return score.score


def calculate_bleu_scores(reference_translations: List[str], 
                         predictions: dict) -> dict:
    """
    Calculate BLEU scores for multiple models.
    
    Args:
        reference_translations: List of reference translations
        predictions: Dictionary mapping model names to their predictions
    
    Returns:
        Dictionary mapping model names to BLEU scores
    """
    results = {}
    
    for model_name, model_predictions in predictions.items():
        bleu_score = calculate_bleu(reference_translations, model_predictions)
        results[model_name] = bleu_score
    
    return results
