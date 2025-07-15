# Import libraries
import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class MicrobiomeAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=32):
        super(MicrobiomeAutoencoder,self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,latent_dim)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim,64),
            nn.ReLU(),
            nn.Linear(64,128),
            nn.ReLU(),
            nn.Linear(128,input_dim),
            nn.Sigmoid()
        )

    def forward(self,x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon

    def get_latent(self, x):
        return self.encoder(x)
    

def train_autoencoder(model, data_tensor, epochs=100, batch_size=64, lr=1e-3, device='cuda'):
    print("Training autoencoder...")
    model.to(device)
    model.train()

    dataset = TensorDataset(data_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            x = batch[0].to(device)
            optimizer.zero_grad()
            x_recon = model(x)
            loss = criterion(x_recon, x)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x.size(0)

        avg_loss = epoch_loss / len(dataset)
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
