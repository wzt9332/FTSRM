import torch
import numpy as np

def naive_sparse_bmm(sparse_mat, dense_mat, transpose=False):
    if transpose:
        return torch.stack([torch.sparse.mm(s_mat, d_mat.t()) for s_mat, d_mat in zip(sparse_mat, dense_mat)], 0)
    else:
        return torch.stack([torch.sparse.mm(s_mat, d_mat) for s_mat, d_mat in zip(sparse_mat, dense_mat)], 0)

def sparse_permute(sparse_mat, order):
    values = sparse_mat.coalesce().values()
    indices = sparse_mat.coalesce().indices()
    indices = torch.stack([indices[o] for o in order], 0).contiguous()
    return torch.sparse_coo_tensor(indices, values)





def similarity(a, b):
    L = torch.tensor(2)
    pi = torch.tensor(3.1415926)
    one = torch.tensor(1)
    out = torch.sqrt(L * pi)*(1 - torch.min(a/b, b/a))
    out_1 = (torch.min(a/b, one)*torch.min(a/b, one)) + (torch.min(b/a, one)*torch.min(b/a, one))
    out = out / (torch.sqrt(4-pi) * torch.sqrt(out_1))
    return out




def affinity_mat(Q, img, map):
    aff_mat = []
    batch = Q.shape[0]
    for b in range(batch):
        map_uniq = map[b].unique()
        n_superpixel = len(map_uniq)
        aff_mat_i = torch.tensor(np.zeros((n_superpixel, n_superpixel)))
        for i in range(n_superpixel):
            x, y = torch.where(map[0, 0] == i)
            v_alpha = img[b, x, y].mean()
            for j in range(i, n_superpixel):
                x, y = torch.where(map[0, 0] == j)
                v_beta = img[b, x, y].mean()
                if i == j:
                    aff_mat_i[i, j] = 1
                else:
                    s = similarity(v_alpha, v_beta)
                    aff_mat_i[i, j] = torch.exp(-s)

        aff_mat_i = aff_mat_i + aff_mat_i.T - torch.diag(aff_mat_i.diagonal())
        aff_mat.append(aff_mat_i)

    return aff_mat


def affinity_mat_old(Q, img):
    n_superpixel = Q.shape[1]
    batch = Q.shape[0]
    aff_mat = torch.tensor(np.zeros((batch, n_superpixel, n_superpixel)))
    qv = torch.sum(Q, dim=2).unsqueeze(2)
    v = torch.sum(Q*img,dim=2).unsqueeze(2) / qv

    # vv =
    # v_ = v/vv

    for b in range(batch):
        for i in range(n_superpixel):
            for j in range(i, n_superpixel):
                if i == j:
                    aff_mat[b, i, j] = 1
                else:
                    s = similarity(v[b, i], v[b, j], qv[b, i], qv[b, j])
                    aff_mat[b, i, j] = torch.exp(-s)

        aff_mat[b] = aff_mat[b] + aff_mat[b].T - torch.diag(aff_mat[b].diagonal())

    return aff_mat


# def matadd(mat0, mat1):
#     mat = torch.zeros_like(mat0)
#     for i in range(mat.shape[1]):
#         for j in range(mat.shape[2]):
#             mat[:, i, j] = torch.sum(mat0[:, i, :] * mat1[:, :, j], dim=1)
#
#     return mat


def matmul(mat0, mat1):
    mat = torch.matmul(mat0, mat1)
    return mat





def matrixWalk(Matrix,n):
    if(n==1):
        return Matrix
    else:
        return matmul(Matrix,matrixWalk(Matrix,n-1))





