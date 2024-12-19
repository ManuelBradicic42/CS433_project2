import numpy as np
from typing import List, Union

def beam_search_decoder(predictions: np.ndarray, chars: Union[str, List[str]], beam_width: int = 3) -> List[tuple]:
    """
    CTC beam search decoder.

    Args:
        predictions (np.ndarray): Logits or probabilities from the model (T, C).
        chars (Union[str, List[str]]): List of characters corresponding to classes.
        beam_width (int): The number of beams to keep.

    Returns:
        List[tuple]: List of (decoded_text, score) tuples.
    """
    T, C = predictions.shape
    beams = [("", 0.0)]  # list of tuples (sequence, score)

    for t in range(T):
        new_beams = []
        for seq, score in beams:
            for c in range(C):
                char = chars[c] if c < len(chars) else ''
                new_seq = seq + char
                new_score = score + np.log(predictions[t, c] + 1e-8)  # Avoid log(0)
                new_beams.append((new_seq, new_score))
        
        # Keep top beam_width beams
        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
    
    # Normalize scores and return
    total_scores = np.array([score for _, score in beams])
    total_scores -= np.max(total_scores)  # For numerical stability
    probabilities = np.exp(total_scores)
    probabilities /= np.sum(probabilities)
    
    return [(seq, prob) for seq, prob in zip([b[0] for b in beams], probabilities)]