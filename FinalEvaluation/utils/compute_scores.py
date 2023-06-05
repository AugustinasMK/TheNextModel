import torch


def compute_scores(emb_one, emb_two):
    """Computes cosine similarity between two vectors."""
    scores = torch.matmul(emb_one, emb_two)
    return scores.cpu().numpy().tolist()
