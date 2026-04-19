import torch
import torch.optim as optim
from nca.model import NCA
from nca.utils import normalize
from nca.corruption import corrupt

def train_model(data, epochs=400):
    model = NCA()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    data_norm = normalize(data)

    for epoch in range(epochs):
        optimizer.zero_grad()

        noisy, mask = corrupt(data_norm)
        output = model(noisy, mask)

        loss = ((output - data_norm) ** 2 * mask).mean()

        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    return model