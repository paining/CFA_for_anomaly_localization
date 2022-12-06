import random
import argparse

import torch
from torch.utils.data import DataLoader

from cnn.resnet import wide_resnet50_2 as wrn50_2
from cnn.resnet import resnet18 as res18
from cnn.efficientnet import EfficientNet as effnet
from cnn.vgg import vgg19_bn as vgg19

import datasets.mvtec as mvtec
from datasets.mvtec import MVTecDataset
from datasets.mvtec_finetuning import MVTecAnomalyDataset
from utils.metric import *
from utils.visualizer import *

from utils.cfa import *
import torch.optim as optim
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


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

    return parser.parse_args()


def run():
    seed = 1024
    random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)

    args = parse_args()
    class_names = (
        mvtec.CLASS_NAMES if args.class_name == "all" else [args.class_name]
    )

    total_roc_auc = []
    total_pixel_roc_auc = []
    total_pixel_pro_auc = []

    fig_loss = plt.figure("loss")
    ax_loss_train = fig_loss.subplots(1, 1)
    ax_loss_train.set_yscale("log")
    ax_loss_val = ax_loss_train.twinx()
    ax_loss_val.set_yscale("log")
    fig = plt.figure("AUROC", figsize=(20, 10))
    ax = fig.subplots(1, 2)
    fig_img_rocauc = ax[0]
    fig_pixel_rocauc = ax[1]

    for class_name in class_names:
        best_img_roc = -1
        best_pxl_roc = -1
        best_pxl_pro = -1
        training_loss = []
        eval_loss = []
        save_path = os.path.join(args.save_path, class_name)
        os.makedirs(save_path, exist_ok=True)
        print(" ")
        print("%s | newly initialized..." % class_name)

        train_dataset = MVTecDataset(
            dataset_path=args.data_path,
            class_name=class_name,
            resize=256,
            cropsize=args.size,
            is_train=True,
            wild_ver=args.Rd,
        )

        if args.anomaly_set:
            anomaly_dataset = MVTecAnomalyDataset(
                dataset_path=args.data_path,
                class_name=class_name,
                anomaly_set=args.anomaly_set,
                resize=256,
                cropsize=args.size,
                is_train=True,
                wild_ver=args.Rd,
            )

        test_dataset = MVTecDataset(
            dataset_path=args.data_path,
            class_name=class_name,
            resize=256,
            cropsize=args.size,
            is_train=False,
            wild_ver=args.Rd,
        )

        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=4,
            pin_memory=True,
            shuffle=True,
            drop_last=True,
        )

        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=4,
            pin_memory=True,
        )

        if args.cnn == "wrn50_2":
            model = wrn50_2(pretrained=True, progress=True)
        elif args.cnn == "res18":
            model = res18(pretrained=True, progress=True)
        elif args.cnn == "effnet-b5":
            model = effnet.from_pretrained("efficientnet-b5")
        elif args.cnn == "vgg19":
            model = vgg19(pretrained=True, progress=True)

        model = model.to(device)
        model.eval()

        loss_fn = DSVDD(
            model, train_loader, args.cnn, args.gamma_c, args.gamma_d, device
        )
        if args.pretrained:
            loss_fn.load_state_dict(torch.load(args.pretrained))
        loss_fn = loss_fn.to(device)

        epochs = 30
        params = [
            {"params": loss_fn.parameters()},
        ]
        optimizer = optim.AdamW(
            params=params, lr=1e-3, weight_decay=5e-4, amsgrad=True
        )

        for epoch in tqdm(range(epochs), "%s -->" % (class_name)):
            r"TEST PHASE"

            test_imgs = list()
            gt_mask_list = list()
            gt_list = list()
            heatmaps = None

            loss_fn.train()
            losses = []
            for (x, _, _) in tqdm(train_loader, desc="training", leave=False):
                optimizer.zero_grad()
                p = model(x.to(device))

                loss, _ = loss_fn(p)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            training_loss.append(np.mean(losses).item())
            ax_loss_train.clear()
            ax_loss_val.clear()
            ax_loss_train.plot(training_loss, '-r', label="training")

            loss_fn.eval()
            for x, y, mask in tqdm(test_loader, desc="evaluation", leave=False):
                test_imgs.extend(x.cpu().detach().numpy())
                gt_list.extend(y.cpu().detach().numpy())
                gt_mask_list.extend(mask.cpu().detach().numpy())

                p = model(x.to(device))
                _, score = loss_fn(p)
                heatmap = score.cpu().detach()
                heatmap = torch.mean(heatmap, dim=1)
                heatmaps = (
                    torch.cat((heatmaps, heatmap), dim=0)
                    if heatmaps != None
                    else heatmap
                )
            eval_loss.append(torch.mean(heatmaps).item())
            ax_loss_val.plot(eval_loss, '-b', label="eval")
            fig_loss.legend()
            fig_loss.savefig(os.path.join(save_path, "loss.png"), dpi=100)

            heatmaps = upsample(heatmaps, size=x.size(2), mode="bilinear")
            heatmaps = gaussian_smooth(heatmaps, sigma=4)

            gt_mask = np.asarray(gt_mask_list)
            scores = rescale(heatmaps)

            scores = scores
            threshold = get_threshold(gt_mask, scores)

            r"Image-level AUROC"
            fpr, tpr, img_roc_auc = cal_img_roc(scores, gt_list)
            # best_img_roc = img_roc_auc if img_roc_auc > best_img_roc else best_img_roc
            if img_roc_auc > best_img_roc:
                best_img_roc = img_roc_auc
                best_img_fpr = fpr
                best_img_tpr = tpr

                torch.save(
                    loss_fn.state_dict(), 
                    os.path.join(save_path, f"{class_name}_best.pt")
                )

            # fig_img_rocauc.plot(fpr, tpr, label='%s img_ROCAUC: %.3f' % (class_name, img_roc_auc))

            r"Pixel-level AUROC"
            fpr, tpr, per_pixel_rocauc = cal_pxl_roc(gt_mask, scores)
            # best_pxl_roc = per_pixel_rocauc if per_pixel_rocauc > best_pxl_roc else best_pxl_roc
            if per_pixel_rocauc > best_pxl_roc:
                best_pxl_roc = per_pixel_rocauc
                best_pxl_fpr = fpr
                best_pxl_tpr = tpr

            r"Pixel-level AUPRO"
            per_pixel_proauc = cal_pxl_pro(gt_mask, scores)
            best_pxl_pro = (
                per_pixel_proauc
                if per_pixel_proauc > best_pxl_pro
                else best_pxl_pro
            )

            tqdm.write(
                "[%d / %d]image ROCAUC: %.3f | best: %.3f"
                % (epoch, epochs, img_roc_auc, best_img_roc)
            )
            tqdm.write(
                "[%d / %d]pixel ROCAUC: %.3f | best: %.3f"
                % (epoch, epochs, per_pixel_rocauc, best_pxl_roc)
            )
            tqdm.write(
                "[%d / %d]pixel PROAUC: %.3f | best: %.3f"
                % (epoch, epochs, per_pixel_proauc, best_pxl_pro)
            )

        print("image ROCAUC: %.3f" % (best_img_roc))
        print("pixel ROCAUC: %.3f" % (best_pxl_roc))
        print("pixel ROCAUC: %.3f" % (best_pxl_pro))

        total_roc_auc.append(best_img_roc)
        total_pixel_roc_auc.append(best_pxl_roc)
        total_pixel_pro_auc.append(best_pxl_pro)

        fig_img_rocauc.plot(
            best_img_fpr,
            best_img_tpr,
            label="%s img_ROCAUC: %.3f" % (class_name, img_roc_auc),
        )
        fig_pixel_rocauc.plot(
            best_pxl_fpr,
            best_pxl_tpr,
            label="%s ROCAUC: %.3f" % (class_name, per_pixel_rocauc),
        )
        save_dir = args.save_path + "/" + f"pictures_{args.cnn}"
        os.makedirs(save_dir, exist_ok=True)
        plot_fig(
            test_imgs, scores, gt_mask_list, threshold, save_dir, class_name
        )

    print("Average ROCAUC: %.3f" % np.mean(total_roc_auc))
    fig_img_rocauc.title.set_text(
        "Average image ROCAUC: %.3f" % np.mean(total_roc_auc)
    )
    fig_img_rocauc.legend(loc="lower right")

    print("Average pixel ROCUAC: %.3f" % np.mean(total_pixel_roc_auc))
    fig_pixel_rocauc.title.set_text(
        "Average pixel ROCAUC: %.3f" % np.mean(total_pixel_roc_auc)
    )
    fig_pixel_rocauc.legend(loc="lower right")

    print("Average pixel PROUAC: %.3f" % np.mean(total_pixel_pro_auc))

    fig.tight_layout()
    fig.savefig(os.path.join(args.save_path, "roc_curve.png"), dpi=100)


if __name__ == "__main__":
    run()
