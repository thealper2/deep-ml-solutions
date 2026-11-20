import torch


def model_policy(model, temperature=1.0, generator=None, max_new_tokens=8):
    """Return policy(transcript) -> the model's next action as text, generated token by token."""
    def policy(transcript):
        tokens = transcript.split()
        ids = [STOI.get(t, STOI["<unk>"]) for t in tokens]
        generated = []

        with torch.no_grad():
            for _ in range(max_new_tokens):
                ctx = ids[-model.block_size:]
                x = torch.tensor([ctx], dtype=torch.long)
                logits = model(x)
                last = logits[0, -1, :]

                if temperature == 0:
                    nxt = int(torch.argmax(last).item())
                else:
                    probs = F.softmax(last / temperature, dim=-1)
                    nxt = int(torch.multinomial(probs, 1, generator=generator).item())

                tok = VOCAB[nxt]
                ids.append(nxt)
                generated.append(tok)
                if tok in ("</call>", "</answer>"):
                    break

        return " ".join(generated)
    return policy