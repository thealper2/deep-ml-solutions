import numpy as np

def resize_lm_head(tok_emb, out_head, new_vocab_size, init_value=0.0):
    """Resize a token embedding matrix and an output head matrix to a new vocab size.

    Args:
        tok_emb: array-like of shape (V, d)
        out_head: array-like of shape (V, d)
        new_vocab_size: int, target vocabulary size V_new (must be >= V)
        init_value: float, fill value for newly appended rows
    Returns:
        list: [new_tok_emb, new_out_head] as 2D Python lists
    """
    tok_emb = np.array(tok_emb)
    out_head = np.array(out_head)
    V, d = tok_emb.shape
    rows_to_add = new_vocab_size - V
    new_rows = np.full((rows_to_add, d), init_value)
    new_tok_emb = np.vstack([tok_emb, new_rows])
    new_out_head = np.vstack([out_head, new_rows])
    return [new_tok_emb.tolist(), new_out_head.tolist()]
