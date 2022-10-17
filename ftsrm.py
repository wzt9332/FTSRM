import torch
from .sparse_utils import naive_sparse_bmm, sparse_permute
import numpy as np
from torch.functional import F



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




def similarity2(a, b, qa, qb):
    L = 2
    pi = 3.1415926
    out = np.sqrt(L * pi)*(1 - min(a/b, b/a))
    out_1 = (min(a/b, 1)*min(a/b, 1))/qa + (min(b/a, 1)*min(b/a, 1))/qb
    out = out / (np.sqrt(4-pi) * np.sqrt(out_1))
    out = np.exp(-out)
    return out



def comp_merge_loss(Q, img, map, labels, device):
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



