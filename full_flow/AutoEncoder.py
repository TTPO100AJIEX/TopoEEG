import os
import random

import tqdm
import torch
import numpy
import matplotlib.pyplot as plt
import torch.utils.data


device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


def set_random_seed(seed: int):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True


class BaseAutoEncoder:
    def __init__(
        self,
        n_features: int,
        n_components: int,

        batch_size: int = 64,
        learning_rate: float = 1e-3,
        n_epochs: int = 250,
        random_state: int = 42
    ):
        set_random_seed(random_state)

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(in_features = n_features, out_features = 4000), torch.nn.BatchNorm1d(4000), torch.nn.ReLU(),
            torch.nn.Linear(in_features = 4000, out_features = 1000), torch.nn.BatchNorm1d(1000), torch.nn.ReLU(),
            torch.nn.Linear(in_features = 1000, out_features = n_components)
        ).to(device)
        self.decoder = torch.nn.Linear(in_features = n_components, out_features = n_features).to(device)
        
        self.n_components = n_components
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.random_state = random_state
    
    def loss(self, data: torch.Tensor, encoded: torch.Tensor, decoded: torch.Tensor):
        raise NotImplementedError
    
    def fit(self, data: torch.Tensor):
        def lr_scheduler(epoch: int):
            if epoch < self.n_epochs * 1 // 3: return 1
            if epoch < self.n_epochs * 2 // 3: return 0.1
            return 0.01
        
        dataloader = torch.utils.data.DataLoader(data, batch_size = self.batch_size, shuffle = True)

        optimizer = torch.optim.AdamW(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr = self.learning_rate
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scheduler)
        
        self.encoder.train()
        self.decoder.train()
        for _ in tqdm.trange(self.n_epochs, desc = f"{self.n_components}"):
            for data in dataloader:
                optimizer.zero_grad()
                encoded = self.encoder(data)
                decoded = self.decoder(encoded)

                loss = self.loss(data, encoded, decoded)
                loss.backward()
                optimizer.step()
                scheduler.step()
        
    
    def fit_transform(self, features: numpy.ndarray):
        set_random_seed(self.random_state)
        data = torch.Tensor(features).to(device)
        self.fit(data)
        self.encoder.eval()
        return self.encoder(data).detach().cpu().numpy()

class AutoEncoder(BaseAutoEncoder):
    def loss(self, data: torch.Tensor, encoded: torch.Tensor, decoded: torch.Tensor):
        return torch.nn.functional.mse_loss(decoded, data, reduction = 'mean')
