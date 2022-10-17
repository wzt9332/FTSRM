import math
import os

import numpy as np
import torch
import cv2

from lib.ssn.ssn import ssn_iter, sparse_ssn_iter
from skimage.segmentation._slic import _enforce_label_connectivity_cython

from lib.ssn.ssn import sparse_ssn_iter


@torch.no_grad()
def inference(image, niter, nspix, enforce_connectivity=True):
    """
    generate superpixels

    Args:
        image: numpy.ndarray
            An array of shape (h, w, c)
        nspix: int
            number of superpixels
        n_iter: int
            number of iterations
        fdim (optional): int
            feature dimension for supervised setting
        color_scale: float
            color channel factor
        pos_scale: float
            pixel coordinate factor
        weight: state_dict
            pretrained weight
        enforce_connectivity: bool
            if True, enforce superpixel connectivity in postprocessing

    Return:
        labels: numpy.ndarray
            An array of shape (h, w)
    """


    height, width = image.shape[:2]



    coords = torch.stack(torch.meshgrid(torch.arange(height, device="cuda"), torch.arange(width, device="cuda")), 0)
    coords = coords[None].float()

    image = torch.from_numpy(image)
    image = image.permute(2, 0, 1).float()
    image = image[0, :, :].unsqueeze(0).unsqueeze(0).cuda()

    inputs = torch.cat([image, coords], 1)


    Q_ = model(inputs)
    Q, H, feat = ssn_iter(Q_, nspix, niter)
    print(len(torch.unique(H)))

    labels = H.reshape(height, width).to("cpu").detach().numpy()

    if enforce_connectivity:
        segment_size = height * width / nspix
        min_size = int(0.06 * segment_size)
        max_size = int(3.0 * segment_size)
        labels = _enforce_label_connectivity_cython(
            labels[None], min_size, max_size)[0]

    return labels


if __name__ == "__main__":
    import time
    import argparse
    import matplotlib.pyplot as plt
    from skimage.segmentation import mark_boundaries
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", default='./test_img', type=str, help="/path/to/image")
    parser.add_argument("--weight", default='./log/dssg_suppix_15000.pth', type=str, help="/path/to/pretrained_weight")
    parser.add_argument("--fdim", default=20, type=int, help="embedding dimension")
    parser.add_argument("--niter", default=10, type=int, help="number of iterations for differentiable SLIC")
    parser.add_argument("--nspix", default=100, type=int, help="number of superpixels")
    parser.add_argument("--color_scale", default=0.26, type=float)
    parser.add_argument("--pos_scale", default=2.5, type=float)
    args = parser.parse_args()

    img_list = os.listdir(args.image_path)


    from model import dssg_pix_net
    model = dssg_pix_net(args.nspix).to('cuda')
    model.load_state_dict(torch.load(args.weight))
    model.eval()

    for name in img_list:

        image = cv2.imread(os.path.join(args.image_path, name), cv2.IMREAD_UNCHANGED)

        s = time.time()
        label = inference(image, args.niter, args.nspix)
        print(f"time {time.time() - s}sec")
        plt.imsave(name.split('.')[0]+".png", mark_boundaries(image, label))
