import argparse
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from rich.console import Console

from sklearn.manifold import TSNE

from utils.cfa import DSVDD as CFA

from cnn.resnet import wide_resnet50_2 as wrn50_2

import datasets.mvtec as mvtec
from datasets.mvtec import MVTecDataset
import matplotlib as mpl
import matplotlib.pyplot as plt
import umap.umap_ as umap
import pickle

console = Console()

def parse_args():
    parser = argparse.ArgumentParser("CFA configuration")
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--save_path", type=str, default="./mvtec_result")
    parser.add_argument("--Rd", type=bool, default=False)
    parser.add_argument(
        "--cnn",
        type=str,
        choices=["res18", "wrn50_2", "effnet-b5", "vgg19"],
        default="wrn50_2",
    )
    parser.add_argument("--size", type=int, choices=[224, 256], default=224)
    parser.add_argument("--gamma_c", type=int, default=1)
    parser.add_argument("--gamma_d", type=int, default=1)

    parser.add_argument("--class_name", type=str, default="all")
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--anomaly_set", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--eval", action="store_true")

    return parser.parse_args()

args = parse_args()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

train_dataset = MVTecDataset(
    dataset_path=args.data_path,
    class_name=args.class_name,
    resize=256,
    cropsize=args.size,
    is_train=True,
    wild_ver=args.Rd,
)
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=4,
    pin_memory=True,
    shuffle=True,
    drop_last=True,
)
if args.anomaly_set:
    test_dataset = MVTecDataset(
        dataset_path=args.data_path,
        class_name=args.class_name,
        resize=256,
        cropsize=args.size,
        is_train=False,
        wild_ver=args.Rd,
        anomaly_subset=[args.anomaly_set]
    )
else:
    test_dataset = MVTecDataset(
        dataset_path=args.data_path,
        class_name=args.class_name,
        resize=256,
        cropsize=args.size,
        is_train=False,
        wild_ver=args.Rd,
    )
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=4,
    pin_memory=True,
)


model = wrn50_2(pretrained=True, progress=True)
model = model.to(device)
model.eval()

loss_fn = CFA(
    model, train_loader, args.cnn, args.gamma_c, args.gamma_d, device
)
if args.pretrained:
    print("loading pretrained model : ", args.pretrained)
    loss_fn.load_state_dict(torch.load(args.pretrained))
loss_fn = loss_fn.to(device)

centroid = loss_fn.C.cpu().detach().numpy()


features = []
labels = []
loss_fn.eval()
for x, y, mask in tqdm(test_loader, desc="evaluation", leave=False):
    p = model(x.to(device))

    feature = loss_fn.Descriptor(p)
    mask = F.interpolate(mask, feature.shape[-2:], mode="nearest")

    feature = rearrange(feature, "b c h w -> (b h w) c")
    feature = feature.cpu().detach().numpy()
    mask = mask.to(torch.int32)*y.reshape(-1,1,1,1).expand(mask.shape)
    mask = rearrange(mask, "b c h w -> (b h w) c")
    mask = mask.cpu().detach().numpy()
    features.append(feature)
    labels.append(mask)

features = np.concatenate(features, axis=0)
labels = np.concatenate(labels, axis=0)
#manifold = TSNE(verbose=1, n_jobs=8)
manifold = umap.UMAP(
    n_neighbors=20, n_components=2, metric="euclidean", verbose=True
)
#with console.status(f"Manifold by {manifold.__class__}", spinner="dots"):
manifold_features = manifold.fit_transform(features)


with open("backup.pkl", "w") as f:
    pickle.dump({"manifold_features":manifold_features, "labels":labels}, f)
fig, subplot = plt.subplots(1,1)

subplot.scatter(manifold_features[:,0], manifold_features[:,1], cmap=mpl.colormaps['jet'])