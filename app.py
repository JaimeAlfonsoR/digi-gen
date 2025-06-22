import streamlit as st
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import io

# VAE model (same structure as training)
class VAE(torch.nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.fc1 = torch.nn.Linear(784, 400)
        self.fc21 = torch.nn.Linear(400, 20)
        self.fc22 = torch.nn.Linear(400, 20)
        self.fc3 = torch.nn.Linear(20, 400)
        self.fc4 = torch.nn.Linear(400, 784)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc21(h), self.fc22(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE().to(device)
model.load_state_dict(torch.load("vae_mnist.pth", map_location=device))
model.eval()

# Streamlit UI
st.title("MNIST Digit Generator")
digit = st.selectbox("Choose a digit (for display only, no conditioning):", list(range(10)))

if st.button("Generate"):
    images = []
    with torch.no_grad():
        for _ in range(5):
            z = torch.randn(1, 20).to(device)
            sample = model.decode(z).cpu().view(28, 28)
            images.append(sample)

    # Display
    fig, axs = plt.subplots(1, 5, figsize=(10, 2))
    for i, img in enumerate(images):
        axs[i].imshow(img, cmap='gray')
        axs[i].axis('off')
    st.pyplot(fig)
