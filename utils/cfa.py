import torch
import torch.nn as nn
from einops import rearrange
from tqdm import tqdm
from sklearn.cluster import KMeans
from .metric import *
from utils.coordconv import CoordConv2d
import torch.nn.functional as F
from utils.coreset_sampling import coreset_subsampling_gpu


class DSVDD(nn.Module):
    def __init__(self, model, cnn, gamma_c, gamma_d, device):
        super(DSVDD, self).__init__()
        self.device = device

        self.C = torch.zeros((1792, 3136), device=device)
        self.nu = 1e-3
        self.scale = None

        self.gamma_c = gamma_c
        self.gamma_d = gamma_d
        self.alpha = 1e-1
        self.K = 3
        self.J = 3

        self.r = nn.Parameter(1e-5 * torch.ones(1), requires_grad=True)
        self.Descriptor = Descriptor(self.gamma_d, cnn).to(device)
        # self._init_centroid(model, data_loader)  # (1, C, H, W)
        # # Memory Bank Size(M) = Number of Images * H * W
        # self.C = rearrange(self.C, "b c h w -> (b h w) c").detach()  # (M, C)

        # if self.gamma_c > 1:
        #     self.C = self.C.cpu().detach().numpy()
        #     n_clusters = self.C.shape[0] // self.gamma_c
        #     print("Number of Memory Bank Clusters =", n_clusters)
        #     self.C = (
        #         KMeans(n_clusters=n_clusters, max_iter=3000, verbose=1)
        #         .fit(self.C)
        #         .cluster_centers_
        #     )
        #     self.C = torch.Tensor(self.C).to(device)

        # self.C = self.C.transpose(-1, -2).detach()  # (C, M) = (C, H*W)
        self.C = nn.Parameter(self.C, requires_grad=False)
        self.scale = 56

        print("Initialize Model Done.")

    def forward(self, p, mask=None):
        phi_p = self.Descriptor(p)
        phi_p = rearrange(phi_p, "b c h w -> b (h w) c")

        score = 0
        loss = 0
        if self.training:
            loss = self._soft_boundary(phi_p, mask)
        else:
            features = torch.sum(torch.pow(phi_p, 2), 2, keepdim=True)
            centers = torch.sum(torch.pow(self.C, 2), 0, keepdim=True)
            f_c = 2 * torch.matmul(phi_p, (self.C))
            # (B, H*W, M) = (B, H*W, 1) + (1, M) - (B, H*W, M)
            dist = features + centers - f_c
            dist = torch.sqrt(dist)

            n_neighbors = self.K
            dist = dist.topk(n_neighbors, largest=False).values

            dist = (F.softmin(dist, dim=-1)[:, :, 0]) * dist[:, :, 0]
            dist = dist.unsqueeze(-1)  # (B, H*W)

            score = rearrange(dist, "b (h w) c -> b c h w", h=self.scale)

        return loss, score

    def _soft_boundary(self, phi_p: torch.Tensor, mask):
        """
        phi_p : input patch-wise feature (B, H*W, C)

        """
        if mask == None:
            mask = torch.zeros(
                *phi_p.shape[:2], dtype=torch.bool, device=phi_p.device
            )
        else:
            mask = rearrange(mask, "b c h w -> b (h w) c")
        mask = mask.squeeze(-1)
        label = mask.sum(dim=1, keepdim=True).to(torch.bool).expand_as(mask)
        # (B, H*W, 1)
        features = torch.sum(torch.pow(phi_p, 2), 2, keepdim=True)
        # (1, M)
        centers = torch.sum(torch.pow(self.C, 2), 0, keepdim=True)
        # (B, H*W, C)x(C, M) = (B, H*W, M)
        f_c = 2 * torch.matmul(phi_p, (self.C))
        # (B, H*W, M) = (B, H*W, 1) + (1, M) - (B, H*W, M)
        dist = features + centers - f_c
        n_neighbors = self.K + self.J
        dist = dist.topk(n_neighbors, largest=False).values

        # Top K clusters get closer
        # loss (B, H*W, M) += dist > r^2
        score = dist[:, :, : self.K] - self.r**2
        score = score[torch.logical_and(mask == 0, label == 0)]
        L_att = (1 / self.nu) * torch.mean(
            torch.max(torch.zeros_like(score), score)
        )

        # After J clusters get further away
        # loss += dist < r^2 - alpha
        score = self.r**2 - dist[:, :, self.K :]
        score = score[mask == 0]
        L_rep = (1 / self.nu) * torch.mean(
            torch.max(torch.zeros_like(score), score - self.alpha)
        )

        # loss += dist < (2*r)^2
        score = (2*self.r)**2 - dist
        score = score[mask == 1]
        L_ano = (1 / self.nu) * torch.mean(
            torch.max(torch.zeros_like(score), score - self.alpha)
        ) if len(score) else 0
        score = dist - (2*self.r)**2
        score = score[torch.logical_and(mask == 0, label == 1)]
        L_ano = L_ano + (1 / self.nu) * torch.mean(
            torch.max(torch.zeros_like(score), score - self.alpha)
        ) if len(score) else L_ano

        loss = L_att + L_rep + L_ano

        return loss

    def _init_centroid(self, model, data_loader):
        for i, (x, _, _) in enumerate(tqdm(data_loader, desc="initialize centroid")):
            x = x.to(self.device)
            p = model(x)
            self.scale = p[0].size(2)
            phi_p = self.Descriptor(p)
            self.C = (
                (self.C * i) + torch.mean(phi_p, dim=0, keepdim=True).detach()
            ) / (i + 1)

    def add_centroid(self, model, data_loader, bank_size):
        # # Use K-Means
        # new_C = 0
        # for i, (x, _, _) in enumerate(tqdm(data_loader, desc="initialize centroid")):
        #     x = x.to(self.device)
        #     p = model(x)
        #     self.scale = p[0].size(2)
        #     phi_p = self.Descriptor(p)
        #     new_C = (
        #         (new_C * i) + torch.mean(phi_p, dim=0, keepdim=True).detach()
        #     ) / (i + 1)
        # new_C = rearrange(new_C, "b c h w -> (b h w) c").detach()  # (M, C)
        #
        # new_C = new_C.cpu().detach().numpy()
        # if self.C.nelement() > 0:
        #     old_C = self.C.transpose(-1, -2).cpu().detach().numpy()
        # else:
        #     old_C = np.zeros((0, new_C.shape[1]))
        # new_C = np.concatenate([old_C, new_C], axis=0)
        # print("Number of Memory Bank Clusters =", bank_size)
        # new_C = (
        #     KMeans(n_clusters=bank_size, max_iter=3000, verbose=1)
        #     .fit(new_C)
        #     .cluster_centers_
        # )
        # new_C = torch.Tensor(new_C).to(self.device)

        # Use Coreset
        features = []
        for i, (x, _, _) in enumerate(tqdm(data_loader, desc="initialize centroid")):
            x = x.to(self.device)
            p = model(x)
            self.scale = p[0].size(2)
            phi_p = self.Descriptor(p)
            phi_p = rearrange(phi_p, "b c h w -> (b h w) c").detach()  # (M, C)
            idxs, _ = coreset_subsampling_gpu(phi_p.detach(), bank_size)
            features.append(phi_p[idxs])
        new_C = torch.concat(features, dim=0)
        # new_C = rearrange(new_C, "b c h w -> (b h w) c").detach()  # (M, C)
        idxs, max_distance = coreset_subsampling_gpu(new_C, bank_size)
        new_C = new_C[idxs]

        if self.C.nelement() > 0:
            old_C = self.C.transpose(-1, -2).detach()
        else:
            old_C = torch.zeros((0, new_C.shape[1]), device=self.device)
        new_C = torch.concat([old_C, new_C], dim=0)
        # idxs, max_distance = coreset_subsampling_gpu(new_C, bank_size)
        # new_C = new_C[idxs]

        new_C = new_C.transpose(-1, -2).detach()  # (C, M) = (C, H*W)
        self.C = nn.Parameter(new_C, requires_grad=False)


class Descriptor(nn.Module):
    def __init__(self, gamma_d, cnn):
        super(Descriptor, self).__init__()
        self.cnn = cnn
        if cnn == "wrn50_2":
            dim = 1792
            self.layer = CoordConv2d(dim, dim // gamma_d, 1)
        elif cnn == "res18":
            dim = 448
            self.layer = CoordConv2d(dim, dim // gamma_d, 1)
        elif cnn == "effnet-b5":
            dim = 568
            self.layer = CoordConv2d(dim, 2 * dim // gamma_d, 1)
        elif cnn == "vgg19":
            dim = 1280
            self.layer = CoordConv2d(dim, dim // gamma_d, 1)

    def forward(self, p):
        sample = None
        for o in p:
            o = (
                F.avg_pool2d(o, 3, 1, 1) / o.size(1)
                if self.cnn == "effnet-b5"
                else F.avg_pool2d(o, 3, 1, 1)
            )
            sample = (
                o
                if sample is None
                else torch.cat(
                    (sample, F.interpolate(o, sample.size(2), mode="bilinear")),
                    dim=1,
                )
            )

        phi_p = self.layer(sample)
        return phi_p
