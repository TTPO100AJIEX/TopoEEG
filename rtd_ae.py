import os
import random
import argparse

import numpy
import torch
import torch.utils.data
import pytorch_lightning

from RTD_AE.src.utils import *
from RTD_AE.src.rtd import RTDLoss
from RTD_AE.src.autoencoder import AutoEncoder

parser = argparse.ArgumentParser()

parser.add_argument('subject')
parser.add_argument('experiment')
parser.add_argument('latent_dim', type = int)
parser.add_argument('-me', '--max_epochs', default = 100, type = int)
parser.add_argument('-rs', '--random_state', default = 42, type = int)

args = parser.parse_args()

def set_random_seed(seed: int = args.random_state) -> None:
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
set_random_seed()

config = {
    "max_epochs": args.max_epochs,
    "rtd_every_n_batches": 1,
    "rtd_start_epoch": 0,
    "rtd_l": 1.0, 
    "card": 50,
    "latent_dim": args.latent_dim,
    "n_hidden_layers": 2,
    "hidden_dim": 256,
    "batch_size": 64,
    "engine": "ripser",
    "is_sym": True,
    "lr": 5e-4
}

scaler = FurthestScaler()
flatten = True
geodesic = False

data = numpy.load(f'{args.subject}/{args.experiment}/full_data.npy').astype(numpy.float32)
config['input_dim'] = data.shape[1]
print('Data:', data.shape)

ds = FromNumpyDataset(data, geodesic = geodesic, scaler = scaler, flatten = flatten, n_neighbors = 2)
print('Dataset:', ds[0][0], ds[0][1].shape, ds[0][2])

def collate_with_matrix(samples):
    indicies, data, labels = zip(*samples)
    data, labels = torch.tensor(numpy.asarray(data)), torch.tensor(numpy.asarray(labels))
    if len(data.shape) > 2:
        dist_data = torch.flatten(data, start_dim=1)
    else:
        dist_data = data
    x_dist = torch.cdist(dist_data, dist_data, p=2) / numpy.sqrt(dist_data.shape[1])
    return data, x_dist, labels

def collate_with_matrix_geodesic(samples):
    indicies, data, labels, dist_data = zip(*samples)
    data, labels = torch.tensor(numpy.asarray(data)), torch.tensor(numpy.asarray(labels))
    x_dist = torch.tensor(numpy.asarray(dist_data)[:, indicies])
    return data, x_dist, labels

collate_fn = collate_with_matrix_geodesic if geodesic else collate_with_matrix
val_loader = torch.utils.data.DataLoader(ds, batch_size = config["batch_size"], num_workers = 16, collate_fn = collate_fn)
train_loader = torch.utils.data.DataLoader(ds, batch_size = config["batch_size"], num_workers = 16, collate_fn = collate_fn, shuffle = True)

a, b, c = next(iter(train_loader))
print('Dataloader:', a.shape, b.shape, c.shape)

model = AutoEncoder(
    encoder = get_linear_model(m_type = 'encoder', **config),
    decoder = get_linear_model(m_type = 'decoder', **config),
    RTDLoss = RTDLoss(dim = 1, lp = 1.0,  **config),
    MSELoss = torch.nn.MSELoss(),
    **config
)

trainer = pytorch_lightning.Trainer(gpus = -1, max_epochs = config['max_epochs'], log_every_n_steps = 1, num_sanity_val_steps = 0)
trainer.fit(model, train_loader, val_loader)

latent, labels = get_latent_representations(model, val_loader)
print('Latent', latent.shape)

numpy.save(f'{args.subject}/{args.experiment}/{args.latent_dim}.npy', latent)
