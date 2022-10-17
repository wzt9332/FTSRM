import os, math
import numpy as np
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.functional import F
from lib.utils.meter import Meter
from model import dssg_net
from lib.dataset import bsds, augmentation
from lib.utils.loss import reconstruct_loss_with_cross_etnropy, reconstruct_loss_with_mse
from lib.ssn.ssn import ssn_iter, sparse_ssn_iter


@torch.no_grad()
def eval(model, loader, cfg, device, edges1):
    class ConfMatrix(object):
        def __init__(self, num_classes):
            self.num_classes = num_classes
            self.mat = None

        def update(self, pred, target):
            n = self.num_classes
            if self.mat is None:
                self.mat = torch.zeros((n, n), dtype=torch.int64, device=pred.device)
            with torch.no_grad():
                k = (target >= 0) & (target < n)
                inds = n * target[k].to(torch.int64) + pred[k]
                self.mat += torch.bincount(inds, minlength=n ** 2).reshape(n, n)

        def get_metrics(self):
            h = self.mat.float()
            acc = torch.diag(h).sum() / h.sum()
            iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
            iu[torch.isnan(iu)] = 0.0
            return torch.mean(iu).item(), acc.item()

    model.eval()
    conf_mat = ConfMatrix(cfg.n_class)
    avg_cost = np.zeros((10, 3))
    for data in loader:
        inputs, labels, lgrp = data

        inputs = inputs.to(device)
        labels = labels.to(device)
        lgrp = lgrp.float().to(device)
        labels_org = labels.view(labels.shape[0],labels.shape[1], -1)
        labels = torch.argmax(labels, dim=1)



        height, width = inputs.shape[-2:]

        coords = torch.stack(torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device)), 0)
        coords = coords[None].repeat(inputs.shape[0], 1, 1, 1).float()

        inputs = torch.cat([inputs, coords], 1)

        Q_, seg_pre = model(inputs, lgrp, edges1)

        Q, H, feat = sparse_ssn_iter(Q_, cfg.nspix, cfg.niter)
        H = (-H.min()) + H

        pre_img = torch.zeros_like(labels_org)

        for b in range(1):
            Hb = H[b]
            for pix in torch.unique(Hb):
                ind0 = torch.where(Hb == pix)
                ind0_l = len(ind0[0])
                pre_img[b, :, ind0[0]] = seg_pre[b, :, pix].repeat(1, ind0_l).view(ind0_l, pre_img.shape[1]).permute(1, 0)



        conf_mat.update(pre_img.argmax(1).flatten(), labels.flatten())

    avg_cost[0, 1:] = conf_mat.get_metrics()
    model.train()
    return avg_cost[0, 1]


def update_param(data, model, optimizer, device, cfg, edges_all):
    inputs, labels, lgrp = data
    inputs = inputs.to(device)
    labels = labels.to(device)
    lgrp = lgrp.float().to(device)
    img = inputs.reshape(inputs.shape[0], inputs.shape[1], -1)
    # labels_gt = labels.clone()
    labels = labels.view(labels.shape[0],labels.shape[1], -1)

    height, width = inputs.shape[-2:]


    coords = torch.stack(torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device)), 0)
    coords = coords[None].repeat(inputs.shape[0], 1, 1, 1).float()

    inputs = torch.cat([inputs, coords], 1)

    Q_, seg_pre = model(inputs, lgrp, edges_all)


    Q, H, feat = ssn_iter(Q_, cfg.nspix, cfg.niter)
    for b in range(cfg.batchsize):
        H[b] = (-H[b].min()) + H[b]


    recons_loss = reconstruct_loss_with_mse(Q, img)
    compact_loss = reconstruct_loss_with_mse(Q, coords.reshape(*coords.shape[:2], -1), H)



    # +++++++
    pre_img = torch.zeros_like(labels)
    w_phi = torch.zeros_like(img)
    for b in range(cfg.batchsize):
        Hb = H[b]
        for pix in torch.unique(Hb):
            ind0 = torch.where(Hb == pix)
            ind0_l = len(ind0[0])
            pre_img[b,:,ind0[0]] = seg_pre[b, :, pix].repeat(1, ind0_l).view(ind0_l,pre_img.shape[1]).permute(1, 0)
            region_num = torch.sum(labels[b,:,ind0[0]], dim=-1)
            region_min = torch.min(region_num[torch.nonzero(region_num)]) / ind0_l
            w_phi[b,:,ind0[0]] = region_min * img[b,:,ind0[0]]


    # +++++++
    seg_loss = F.cross_entropy(pre_img, labels)
    recons_b_loss = reconstruct_loss_with_mse(Q, w_phi)
    loss = 0.1 * recons_loss + 0.1 * compact_loss + 0.1 * recons_b_loss + seg_loss
    #loss = seg_loss #+ 0.1 * loss_suppixel

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return {"loss": loss.item(), "reconstruction": recons_loss.item(), "compact": compact_loss.item(),
            "recons_b": recons_b_loss.item(), "seg_loss": seg_loss.item()}


def train(cfg):
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    model = dssg_net(cfg.n_class, cfg.nspix).to(device)

    optimizer = optim.Adam(model.parameters(), cfg.lr)

    # augment = augmentation.Compose([augmentation.RandomHorizontalFlip(), augmentation.RandomScale(), augmentation.RandomCrop()])
    # train_dataset = bsds.DSSG(cfg.root, cfg.n_class, geo_transforms=augment)
    train_dataset = bsds.DSSG(cfg.root, cfg.n_class)
    train_loader = DataLoader(train_dataset, cfg.batchsize, shuffle=True, drop_last=True, num_workers=cfg.nworkers)

    test_dataset = bsds.DSSG(cfg.root, cfg.n_class, split="val")
    test_loader = DataLoader(test_dataset, 1, shuffle=False, drop_last=False)

    meter = Meter()
    iterations = 0
    max_val_asa = 0

    from segraph.segraph import create_graph
    adj_mat = np.arange(cfg.nspix)
    adj_mat = adj_mat.reshape(int(cfg.nspix**0.5), int(cfg.nspix**0.5))
    vertices, edges = create_graph(adj_mat)
    edges = np.array(edges)
    edges = edges[edges[:, 0].argsort()]
    edges = edges.T
    edges1 = edges.copy()
    for i in range(cfg.batchsize):
        if i == 0:
            edges_all = edges
        else:
            edges = edges + cfg.nspix
            edges_all = np.concatenate((edges_all, edges), axis=1)
    edges_all = torch.from_numpy(edges_all).to(device)
    edges1 = torch.from_numpy(edges1).to(device)


    while iterations < cfg.train_iter:
        for data in train_loader:
            iterations += 1
            metric = update_param(data, model, optimizer, device, cfg, edges_all)
            meter.add(metric)
            state = meter.state(f"[{iterations}/{cfg.train_iter}]")
            print(state)
            if (iterations % cfg.test_interval) == 0:
                asa = eval(model, test_loader, cfg, device, edges1)
                print(f"miou {asa}")
                torch.save(model.state_dict(), os.path.join(cfg.out_dir, "model_"+str(iterations)+".pth"))
                if asa > max_val_asa:
                    max_val_asa = asa
                    torch.save(model.state_dict(), os.path.join(cfg.out_dir, "best_model.pth"))
            if iterations == cfg.train_iter:
                break

    unique_id = str(int(time.time()))
    torch.save(model.state_dict(), os.path.join(cfg.out_dir, "model"+unique_id+".pth"))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default='./data/hongkong', type=str, help="/workspace/ssn-pytorch/data/BSR")
    parser.add_argument("--n_class", default=9, type=int, help="class number")
    parser.add_argument("--niter", default=5, type=int, help="number of iterations for differentiable SLIC")
    parser.add_argument("--nspix", default=100, type=int, help="number of superpixels")
    parser.add_argument("--out_dir", default="./log", type=str, help="/path/to/output directory")
    parser.add_argument("--batchsize", default=4, type=int)
    parser.add_argument("--nworkers", default=4, type=int, help="number of threads for CPU parallel")
    parser.add_argument("--lr", default=1e-3, type=float, help="learning rate")
    parser.add_argument("--train_iter", default=100000, type=int)
    parser.add_argument("--fdim", default=20, type=int, help="embedding dimension")
    parser.add_argument("--color_scale", default=0.26, type=float)
    parser.add_argument("--pos_scale", default=2.5, type=float)
    parser.add_argument("--compactness", default=1e-5, type=float)
    parser.add_argument("--test_interval", default=500, type=int)

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    train(args)
