import torch

def evaluate_case(model, reconstruct_fn, samples, mode="normal"):
    errors = []

    for s in samples:
        original = s.copy()

        if mode == "missing":
            s[random.randint(0, 2)] = 0.0
        elif mode == "noisy":
            s[random.randint(0, 2)] *= 10

        pred = reconstruct_fn(model, s)
        true = torch.tensor(original, dtype=torch.float32)

        error = torch.abs(pred - true).mean().item()
        errors.append(error)

    return sum(errors) / len(errors)