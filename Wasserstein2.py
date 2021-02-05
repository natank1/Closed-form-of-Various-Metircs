import torch
import scipy.linalg as lin_alg
import numpy as np

#This function supports batches Neverthelss it iis exposed to symeig issues( to be fixed)
def dist_W2_batch(mean1, cov1, mean2, cov2, index):
    delta_mean = mean1-mean2
    delta_2_norm= torch.pow(torch.norm(delta_mean,dim=index),2)
    e0, v0 = torch.symeig(cov1, eigenvectors=True)
    xxa = torch.diag_embed(torch.sqrt(e0), offset=0, dim1=-2, dim2=-1)
    aa=v0.transpose(-2,-1)
    sqrt_cov1 = torch.matmul(torch.matmul(v0, xxa), aa)
    mm =torch.matmul(torch.matmul(sqrt_cov1,cov2),sqrt_cov1)

    e1, v1 = torch.symeig(mm, eigenvectors=True)

    xxb = torch.diag_embed(torch.sqrt(e1), offset=0, dim1=-2, dim2=-1)
    aa = v1.transpose(-2, -1)
    sqrt_mm = 2*torch.matmul(torch.matmul(v1, xxb), aa)
    ff=torch.diagonal(cov1+cov2-sqrt_mm, dim1=-2, dim2=-1).sum(-1)
    w2=delta_2_norm+ff
    return w2


#For no batches it works great
def dist_W2(mean1, cov1, mean2, cov2, index):
    delta_mean = mean1-mean2
    delta_2_norm= torch.pow(torch.norm(delta_mean,dim=index),2)
    sqrt_cov1= lin_alg.fractional_matrix_power( cov1.detach().numpy(), 0.5)
    mega_mat=  2*torch.tensor(lin_alg.fractional_matrix_power( sqrt_cov1*cov2.detach().numpy()*sqrt_cov1 , 0.5))
    w2=delta_2_norm+torch.trace(cov1+cov2-mega_mat)
    return w2

def dist_W2_torch_diag(mean1, cov1, mean2, cov2,index):
    delta_mean = mean1-mean2
    delta_2_norm= torch.pow(torch.norm(delta_mean),2)

    sqrt_cov1= torch.sqrt( cov1)
    mega_mat=  cov1+cov2-2*torch.sqrt(torch.mul(sqrt_cov1,torch.mul(cov2,sqrt_cov1)))
    w2= delta_2_norm+torch.sum(mega_mat,dim=index)
    return w2



def dist_W2_torch_standard(mean1, cov1, dist_dim,index):
    # Inouts are mean and varaice (not std! ,varaince)
    # lat_dim = mean1.shape[sample_index]
    norm_mu=torch.pow(torch.norm(mean1,dim=index),2)

    trace_of_mega= 2*np.trace( lin_alg.fractional_matrix_power(cov1,0.5))

    w2_score = norm_mu  +torch.trace(cov1)+dist_dim -torch.tensor(trace_of_mega)

    return w2_score


def dist_W2_diag_standard(mean1, cov1, dimension,index):
    norm_mu = torch.pow(torch.norm(mean1,dim=index),2)
    mega_trace=2* torch.sum(torch.sqrt(cov1),dim=index)
    tr1 =torch.sum(cov1,dim=index)
    w2= norm_mu+tr1+dimension -mega_trace

    return w2



