import numpy as np

def clip_score_filter(image_embeds, text_embeds, threshold: float):
    """
    Compute pairwise cosine similarities between image and text embeddings
    (assumed L2-normalized) and zero out entries below `threshold`.

    Returns:
        Nested list of shape (n_images, n_texts).
    """
    image_embeds = np.array(image_embeds)
    text_embeds = np.array(text_embeds)

    if image_embeds.shape[0] == 0 or text_embeds.shape[0] == 0:
        return []

    similarity = image_embeds @ text_embeds.T
    similarity[similarity < threshold] = 0.0
    return similarity.tolist()