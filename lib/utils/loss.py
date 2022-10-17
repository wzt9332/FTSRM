import torch
from .sparse_utils import naive_sparse_bmm, sparse_permute
import numpy as np
from torch.functional import F


def sparse_reconstruction(assignment, labels, hard_assignment=None):
    """
    reconstruction loss with the sparse matrix
    NOTE: this function doesn't use it in this project, because may not return correct gradients

    Args:
        assignment: torch.sparse_coo_tensor
            A Tensor of shape (B, n_spixels, n_pixels)
        labels: torch.Tensor
            A Tensor of shape (B, C, n_pixels)
        hard_assignment: torch.Tensor
            A Tensor of shape (B, n_pixels)
    """
    labels = labels.permute(0, 2, 1).contiguous()

    # matrix product between (n_spixels, n_pixels) and (n_pixels, channels)
    spixel_mean = naive_sparse_bmm(assignment, labels) / (torch.sparse.sum(assignment, 2).to_dense()[..., None] + 1e-16)
    if hard_assignment is None:
        # (B, n_spixels, n_pixels) -> (B, n_pixels, n_spixels)
        permuted_assignment = sparse_permute(assignment, (0, 2, 1))
        # matrix product between (n_pixels, n_spixels) and (n_spixels, channels)
        reconstructed_labels = naive_sparse_bmm(permuted_assignment, spixel_mean)
    else:
        # index sampling
        reconstructed_labels = torch.stack([sm[ha, :] for sm, ha in zip(spixel_mean, hard_assignment)], 0)
    return reconstructed_labels.permute(0, 2, 1).contiguous()


def reconstruction(assignment, labels, hard_assignment=None):
    """
    reconstruction

    Args:
        assignment: torch.Tensor
            A Tensor of shape (B, n_spixels, n_pixels)
        labels: torch.Tensor
            A Tensor of shape (B, C, n_pixels)
        hard_assignment: torch.Tensor
            A Tensor of shape (B, n_pixels)
    """
    labels = labels.permute(0, 2, 1).contiguous()

    # matrix product between (n_spixels, n_pixels) and (n_pixels, channels)
    spixel_mean = torch.bmm(assignment, labels) / (assignment.sum(2, keepdim=True) + 1e-16)
    if hard_assignment is None:
        # (B, n_spixels, n_pixels) -> (B, n_pixels, n_spixels)
        permuted_assignment = assignment.permute(0, 2, 1).contiguous()
        # matrix product between (n_pixels, n_spixels) and (n_spixels, channels)
        reconstructed_labels = torch.bmm(permuted_assignment, spixel_mean)
    else:
        # index sampling
        reconstructed_labels = torch.stack([sm[ha, :] for sm, ha in zip(spixel_mean, hard_assignment)], 0)

    return reconstructed_labels.permute(0, 2, 1).contiguous()


def reconstruct_loss_with_cross_etnropy(assignment, labels, hard_assignment=None):
    """
    reconstruction loss with cross entropy

    Args:
        assignment: torch.Tensor
            A Tensor of shape (B, n_spixels, n_pixels)
        labels: torch.Tensor
            A Tensor of shape (B, C, n_pixels)
        hard_assignment: torch.Tensor
            A Tensor of shape (B, n_pixels)
    """
    reconstracted_labels = reconstruction(assignment, labels, hard_assignment)
    reconstracted_labels = reconstracted_labels / (1e-16 + reconstracted_labels.sum(1, keepdim=True))
    mask = labels > 0
    return -(reconstracted_labels[mask] + 1e-16).log().mean()


def reconstruct_loss_with_mse(assignment, labels, hard_assignment=None):
    """
    reconstruction loss with mse

    Args:
        assignment: torch.Tensor
            A Tensor of shape (B, n_spixels, n_pixels)
        labels: torch.Tensor
            A Tensor of shape (B, C, n_pixels)
        hard_assignment: torch.Tensor
            A Tensor of shape (B, n_pixels)
    """
    reconstracted_labels = reconstruction(assignment, labels, hard_assignment)
    out = torch.nn.functional.mse_loss(reconstracted_labels, labels)

    # out = torch.mean((reconstracted_labels-labels)*(reconstracted_labels-labels))

    return out



def similarity(a, b, n_a, n_b):
    L = torch.tensor(2)
    pi = torch.tensor(3.1415926)
    one = torch.tensor(1)
    out = torch.sqrt(L * pi)*(1 - torch.min(a/b, b/a))
    out_1 = (torch.min(a/b, one)*torch.min(a/b, one))/n_a + (torch.min(b/a, one)*torch.min(b/a, one))/n_b
    out = out / (torch.sqrt(4-pi) * torch.sqrt(out_1))
    return out


def matmul(mat0, mat1):
    mat = torch.matmul(mat0, mat1)
    return mat

def matrixWalk(Matrix,n):
    if(n==1):
        return Matrix
    else:
        return matmul(Matrix,matrixWalk(Matrix,n-1))


def comp_merge_loss(Q, img, map, labels):
    batch = Q.shape[0]
    merge_loss = 0
    for b in range(batch):
        map_uniq = map[b].unique()
        n_superpixel = len(map_uniq)
        aff_mat_i = torch.tensor(np.zeros((n_superpixel, n_superpixel)))
        l_alpha = torch.zeros((n_superpixel, labels.shape[1]), dtype=torch.float64)
        for i in range(n_superpixel):
            x, y = torch.where(map[0, 0] == i)
            v_alpha = img[b, x, y].mean()
            n_alpha = x.shape[0]
            l_pix = labels[b, :, x, y]
            ind = torch.argmax(torch.sum(l_pix, dim=1))
            l_alpha[i, ind] = 1
            for j in range(i, n_superpixel):
                x, y = torch.where(map[0, 0] == j)
                v_beta = img[b, x, y].mean()
                n_beta = x.shape[0]
                if i == j:
                    aff_mat_i[i, j] = 1
                else:
                    s = similarity(v_alpha, v_beta, n_alpha, n_beta)
                    aff_mat_i[i, j] = torch.exp(-s)

        aff_mat_i = aff_mat_i + aff_mat_i.T - torch.diag(aff_mat_i.diagonal())
        walks = 10
        aff_mat_kWalk = matrixWalk(aff_mat_i, walks)
        diag = torch.diagonal(aff_mat_kWalk, 0, 0, 1)
        dim_diag = diag.shape[0]
        diag = diag.repeat(dim_diag).view(dim_diag, dim_diag).permute(1, 0)
        vc = torch.add(aff_mat_kWalk, -diag)
        vc = torch.min(vc, dim=1)[0]
        vc = torch.sigmoid(vc)

        va = vc.repeat(dim_diag).view(dim_diag, dim_diag)
        va = torch.softmax(va * aff_mat_kWalk, dim=1)

        va_s = va / torch.sum(va, dim=0).repeat(dim_diag).view(dim_diag, dim_diag)
        l_beta = torch.matmul(va_s.T, l_alpha)
        merge_loss_i = F.cross_entropy(l_beta, l_alpha, reduction='sum')
        merge_loss = merge_loss + merge_loss_i
        print()

    return merge_loss

def comp_merge_loss2(Q, img, map, labels):
    n_Q = Q.shape[1]
    batch = Q.shape[0]
    aff_mat = torch.tensor(np.zeros((batch, n_Q, n_Q)))

    v_ = img.unsqueeze(1) * Q
    q_sum = torch.sum(torch.sum(Q, dim=2), dim=2)
    q_sum = q_sum.repeat(1, Q.shape[2] * Q.shape[3]).view(Q.shape[0],Q.shape[2],Q.shape[3],Q.shape[1]).permute(0, 3, 1, 2)
    v_ = v_ / q_sum
    v_ = torch.sum(torch.sum(v_,dim=2), dim=2)
    q_sum = q_sum[:,:,0,0]


    for b in range(batch):
        for i in range(n_Q):
            for j in range(i, n_Q):
                if i == j:
                    aff_mat[b, i, j] = 1
                else:
                    s = similarity2(v_[b, i], v_[b, j], q_sum[b, i], q_sum[b, j])
                    aff_mat[b, i, j] = torch.exp(-s)

        aff_mat[b] = aff_mat[b] + aff_mat[b].T - torch.diag(aff_mat[b].diagonal())

    walks = 10
    aff_mat_kWalk = matrixWalk(aff_mat, walks)

    diag = torch.diagonal(aff_mat_kWalk, 0, 1, 2)
    dim_diag = diag.shape[1]
    batch_diag = diag.shape[0]
    diag = diag.repeat(1, dim_diag).view(batch_diag, dim_diag, dim_diag).permute(0, 2, 1)
    vc = torch.add(diag, -aff_mat_kWalk)
    vc = torch.min(vc, dim=2)[0]
    vc = torch.sigmoid(vc)
    va = vc.repeat(1, dim_diag).view(batch_diag, dim_diag, dim_diag)
    va = torch.softmax(va * aff_mat_kWalk, dim=2)

    merge_loss = 0
    for b in range(batch):
        map_uniq = map[b].unique()
        n_superpixel = len(map_uniq)
        l_alpha = torch.zeros((n_superpixel, labels.shape[1]), dtype=torch.float64)
        for i in range(n_superpixel):
            x, y = torch.where(map[0, 0] == i)
            l_pix = labels[b, :, x, y]
            ind = torch.argmax(torch.sum(l_pix, dim=1))
            l_alpha[i, ind] = 1
        va_b = va[b]
        print()


    return



def similarity2(a, b, qa, qb):
    L = 2
    pi = 3.1415926
    out = np.sqrt(L * pi)*(1 - min(a/b, b/a))
    out_1 = (min(a/b, 1)*min(a/b, 1))/qa + (min(b/a, 1)*min(b/a, 1))/qb
    out = out / (np.sqrt(4-pi) * np.sqrt(out_1))
    out = np.exp(-out)
    return out



def comp_merge_loss3(Q, img, map, labels, device):
    merge_loss = 0
    n_superpixel = Q.shape[1]
    batch = Q.shape[0]
    aff_mat = torch.tensor(np.zeros((batch, n_superpixel, n_superpixel))).to(device)
    qv = torch.sum(Q, dim=2).unsqueeze(2)
    v = torch.sum(Q * img, dim=2).unsqueeze(2) / qv


    for b in range(batch):
        l_alpha = torch.zeros((n_superpixel, labels.shape[1]), dtype=torch.float64).to(device)
        for i in range(n_superpixel):
            x = torch.where(map[b] == i)[0]
            l_pix = labels[b][:, x]
            ind = torch.argmax(torch.sum(l_pix, dim=1))
            l_alpha[i, ind] = 1
            for j in range(i, n_superpixel):
                if i == j:
                    aff_mat[b, i, j] = 1
                else:
                    s = similarity2(v[b, i].cpu().detach().numpy(), v[b, j].cpu().detach().numpy(),
                                    qv[b, i].cpu().detach().numpy(), qv[b, j].cpu().detach().numpy())
                    aff_mat[b, i, j] = torch.from_numpy(s)

        aff_mat[b] = aff_mat[b] + aff_mat[b].T - torch.diag(aff_mat[b].diagonal())
        walks = 10
        aff_mat_kWalk = matrixWalk(aff_mat[b], walks)
        diag = torch.diagonal(aff_mat_kWalk, 0)
        dim_diag = diag.shape[0]
        diag = diag.repeat(1, dim_diag).view(dim_diag, dim_diag).permute(1, 0)
        vc = diag - aff_mat_kWalk
        vc = torch.min(vc, dim=1)[0]
        vc = torch.sigmoid(vc)
        va = vc.repeat(1, dim_diag).view(dim_diag, dim_diag)
        va = torch.softmax(va * aff_mat_kWalk, dim=1)
        l_beta = torch.matmul(torch.matmul(va.T,va), l_alpha)
        merge_loss_i = F.cross_entropy(l_beta, l_alpha, reduction='sum')
        merge_loss = merge_loss_i + merge_loss

    return merge_loss


