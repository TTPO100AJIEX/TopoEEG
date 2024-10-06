subj = "Subj3"
exp = "exp_full_flow"

import os

import torch
import numpy

from RTD_AE.src.rtd import RTDLoss

from AutoEncoder import BaseAutoEncoder

class RTDAutoEncoder(BaseAutoEncoder):
    def dist(self, data: torch.Tensor):
        return torch.cdist(data, data) / numpy.sqrt(data.shape[1])

    def loss(self, data: torch.Tensor, encoded: torch.Tensor, decoded: torch.Tensor):
        if not hasattr(self, 'num_iter'):
            self.num_iter = 1
        
        loss = torch.nn.functional.mse_loss(decoded, data, reduction = 'mean')
        if self.num_iter % 5 == 0:
            rtd_loss = RTDLoss(n_threads = -1, engine = "ripser")
            loss_xz, loss_zx, rtd = rtd_loss(self.dist(data), self.dist(encoded))
            loss += rtd
            
        self.num_iter += 1
        return loss

features = numpy.load(f"{subj}/{exp}/qsda/best_features.npy")
TRY_NUM_FEATURES = list(range(10, 201, 10))

for n_components in TRY_NUM_FEATURES:
    folder = f"{subj}/{exp}/features_reduced/rtd_ae/{n_components}"
    os.makedirs(folder, exist_ok = True)

    if os.path.exists(f"{folder}/features.npy"):
        continue
    
    ae = RTDAutoEncoder(n_features = features.shape[1], n_components = n_components)
    numpy.save(f"{folder}/features.npy", ae.fit_transform(features))
