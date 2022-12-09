import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from tqdm import tqdm

CLASS_NAMES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]


class MVTecAnomalyDataset(Dataset):
    def __init__(
        self,
        dataset_path,
        class_name="bottle",
        anomaly_set="broken_small",
        is_train=True,
        is_normal=False,
        resize=256,
        cropsize=224,
        wild_ver=False,
    ):
        assert (
            class_name in CLASS_NAMES
        ), "class_name: {}, should be in {}".format(class_name, CLASS_NAMES)
        assert os.path.exists(
            os.path.join(dataset_path, class_name, "test", anomaly_set)
        ), "there are no {} in {}.".format(anomaly_set, class_name)
        self.dataset_path = dataset_path
        self.class_name = class_name
        self.anomaly_set = anomaly_set
        self.is_train = is_train
        self.resize = resize
        self.cropsize = cropsize
        self.is_normal = is_normal

        self.x, self.y, self.mask = self.load_dataset_folder()

        if wild_ver:
            self.transform_x = T.Compose(
                [
                    T.Resize(resize, Image.ANTIALIAS),
                    T.RandomRotation(10),
                    T.RandomCrop(cropsize),
                    T.ToTensor(),
                    T.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

            self.transform_mask = T.Compose(
                [
                    T.Resize(resize, Image.NEAREST),
                    T.RandomRotation(10),
                    T.RandomCrop(cropsize),
                    T.ToTensor(),
                ]
            )

        else:
            self.transform_x = T.Compose(
                [
                    T.Resize(resize, Image.ANTIALIAS),
                    T.CenterCrop(cropsize),
                    T.ToTensor(),
                    T.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

            self.transform_mask = T.Compose(
                [
                    T.Resize(resize, Image.NEAREST),
                    T.CenterCrop(cropsize),
                    T.ToTensor(),
                ]
            )

    def __getitem__(self, idx):
        x, y, mask = self.x[idx], self.y[idx], self.mask[idx]

        tqdm.write(f"{os.path.relpath(x, self.dataset_path):40}, {y}")
        x = Image.open(x).convert("RGB")
        x = self.transform_x(x)

        if y == 1:
            mask = Image.open(mask)
            mask = self.transform_mask(mask)
        else:
            mask = torch.zeros(1, *x.shape[1:], dtype=torch.bool)

        return x, y, mask

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        x, y, mask = [], [], []

        img_dir = os.path.join(self.dataset_path, self.class_name, "test", self.anomaly_set)
        gt_dir = os.path.join(
            self.dataset_path, self.class_name, "ground_truth", self.anomaly_set
        )

        img_fpath_list = sorted(
            [
                os.path.join(img_dir, f)
                for f in os.listdir(img_dir)
                if f.endswith(".png")
            ]
        )
        x.extend(img_fpath_list)

        if self.is_normal:
            y.extend([0] * len(img_fpath_list))
            mask.extend([None] * len(img_fpath_list))
        else:
            y.extend([1] * len(img_fpath_list))
            img_fname_list = [
                os.path.splitext(os.path.basename(f))[0]
                for f in img_fpath_list
            ]
            gt_fpath_list = [
                os.path.join(gt_dir, img_fname + "_mask.png")
                for img_fname in img_fname_list
            ]
            mask.extend(gt_fpath_list)

        assert len(x) == len(y), "number of x and y should be same"
        assert len(x) == len(mask), "number of x and mask should be same"

        return list(x), list(y), list(mask)
