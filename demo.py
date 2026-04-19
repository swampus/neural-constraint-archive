import torch
from nca.model import NCA
from nca.utils import normalize, denormalize
from nca.data import generate_dataset

def reconstruct(model, items):
    x = torch.tensor([items], dtype=torch.float32)
    x_norm = normalize(x)

    mask = torch.ones_like(x_norm)
    mask[x_norm > 1.5] = 0
    mask[x_norm < 0] = 0

    with torch.no_grad():
        out = model(x_norm, mask)

    return denormalize(out)[0]


def demo():
    model = NCA()
    model.load_state_dict(torch.load("model.pt"))
    model.eval()

    tests = [
        ("Normal", [50.0, 60.0, 70.0]),
        ("Noisy", [50.0, 999.0, 70.0]),
        ("Missing", [50.0, 0.0, 70.0]),
    ]

    for name, data in tests:
        result = reconstruct(model, data)
        print(f"\n{name}")
        print("Input:", data)
        print("Output:", [round(x, 2) for x in result.tolist()])


if __name__ == "__main__":
    demo()