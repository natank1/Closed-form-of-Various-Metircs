# https://djalil.chafai.net/blog/2010/04/30/wasserstein-distance-between-two-gaussians/
import torch
import scipy.linalg as lin_alg
import numpy as np
from torch.distributions import multivariate_normal as gauss
#This function supports batches Neverthelss it iis exposed to symeig issues( to be fixed)


def dist_W2_batch(mean1, cov1, mean2, cov2 ):
    delta_mean = mean1-mean2
    delta_2_norm= torch.pow(torch.norm(delta_mean,dim=-1),2)
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

def covar_frac_power(covar_mat):
    e0, v0 = torch.symeig(covar_mat, eigenvectors=True)
    xxa = torch.diag_embed(torch.sqrt(e0), offset=0, dim1=-2, dim2=-1)
    aa = v0.transpose(-2, -1)
    sqrt_cov1 = torch.matmul(torch.matmul(v0, xxa), aa)
    return sqrt_cov1

def dist_W2_mv_mv(mv_gauss0, mv_gauss1 ):
    # calc delta means
    delta_mean = mv_gauss0.mean-mv_gauss1.mean
    delta_2_norm= torch.pow(torch.norm(delta_mean,dim=-1),2)
    sqrt_cov1= covar_frac_power(mv_gauss0.covariance_matrix)
    mat_prod =torch.matmul(torch.matmul(sqrt_cov1,mv_gauss1.covariance_matrix),sqrt_cov1)
    sqrt_mat2= covar_frac_power(mat_prod)

    sumdi= 2*torch.diagonal(sqrt_mat2,dim1=-2,dim2=-1).sum(-1)

    cov_trace= torch.diagonal(mv_gauss0.covariance_matrix+mv_gauss1.covariance_matrix,dim1=-2,dim2=-1).sum(-1)

    tot_trace =delta_2_norm+cov_trace-sumdi
    return tot_trace


def dist_W2_indepen_mv(diag_gauss,mv_gauss ):
    delta_mean = diag_gauss.mean-mv_gauss.mean
    delta_2_norm= torch.pow(torch.norm(delta_mean,dim=-1),2)

    conv_dimn=torch.unsqueeze(diag_gauss.stddev,dim=-2)
    mat_prod =torch.mul(torch.mul(conv_dimn,mv_gauss.covariance_matrix),conv_dimn)
    sqrt_mm = covar_frac_power(mat_prod)

    # e1, v1 = torch.symeig(mat_prod, eigenvectors=True)
    # xxb = torch.diag_embed(torch.sqrt(e1), offset=0, dim1=-2, dim2=-1)
    # aa = v1.transpose(-2, -1)
    # sqrt_mm= torch.matmul(torch.matmul(v1, xxb), aa)
    sumdi= 2*torch.diagonal(sqrt_mm,dim1=-2,dim2=-1).sum(-1)

    diag_tracce = torch.sum(diag_gauss.variance,dim=-1)

    mv_trace=torch.diagonal(mv_gauss.covariance_matrix,dim1=-2,dim2=-1).sum(-1)

    tot_trace =delta_2_norm+diag_tracce +mv_trace-sumdi
    return tot_trace

def dist_W2_mvGauss(gs0, gs1 ):
    delta_mean = gs0.loc-gs1.loc
    delta_2_norm= torch.pow(torch.norm(delta_mean,dim=-1),2)
    e0, v0 = torch.symeig(gs0.scale, eigenvectors=True)
    xxa = torch.diag_embed(torch.sqrt(e0), offset=0, dim1=-2, dim2=-1)
    aa=v0.transpose(-2,-1)
    sqrt_cov1 = torch.matmul(torch.matmul(v0, xxa), aa)
    mm =torch.matmul(torch.matmul(sqrt_cov1,gs1.scale),sqrt_cov1)

    e1, v1 = torch.symeig(mm, eigenvectors=True)

    xxb = torch.diag_embed(torch.sqrt(e1), offset=0, dim1=-2, dim2=-1)
    aa = v1.transpose(-2, -1)
    sqrt_mm = 2*torch.matmul(torch.matmul(v1, xxb), aa)
    ff=torch.diagonal(gs0.scale+gs1.scale-sqrt_mm, dim1=-2, dim2=-1).sum(-1)
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
    delta_2_norm= torch.pow(torch.norm(delta_mean,dim=index),2)

    sqrt_cov1= torch.sqrt( cov1)
    mega_mat=  cov1+cov2-2*torch.sqrt(torch.mul(sqrt_cov1,torch.mul(cov2,sqrt_cov1)))
    w2= delta_2_norm+torch.sum(mega_mat,dim=index)
    return w2

def dist_W2_diag_0(diag_g0, diag_g1):
    delta_mean = diag_g0.loc- diag_g1.loc
    delta_2_norm= torch.pow(torch.norm(delta_mean,dim=-1),2)

    sqrt_cov1= torch.sqrt( diag_g0.scale)
    mega_mat=  diag_g0.scale+diag_g1.scale-2*torch.sqrt(torch.mul(sqrt_cov1,torch.mul(diag_g1.scale,sqrt_cov1)))
    w2= delta_2_norm+torch.sum(mega_mat,dim=-1)
    return w2


def dist_W2_torch_standard(mean1, cov1, dist_dim,index):
    # Inouts are mean and varaice (not std! ,varaince)
    # lat_dim = mean1.shape[sample_index]
    norm_mu=torch.pow(torch.norm(mean1,dim=index),2)

    trace_of_mega= 2*np.trace( lin_alg.fractional_matrix_power(cov1,0.5))

    w2_score = norm_mu  +torch.trace(cov1)+dist_dim -torch.tensor(trace_of_mega)

    return w2_score


def dist_W2_new_diag_standard(diag_g0 ):
    norm_mu = torch.pow(torch.norm(diag_g0.mean,dim=-1),2)
    mega_trace=2* torch.sum(diag_g0.stddev,dim=-1)


    tr1 =torch.sum(diag_g0.variance,dim=-1)
    w2= norm_mu+tr1+diag_g0.event_shape[0] -mega_trace

    return w2


def dist_W2_diag_1(diag_g0, diag_g1):
    delta_mean = diag_g0.mean- diag_g1.mean
    delta_2_norm= torch.pow(torch.norm(delta_mean,dim=-1),2)

    sqrt_cov1 = diag_g0.stddev

    mega_mat=  diag_g0.variance+diag_g1.variance-2*torch.sqrt(torch.mul(sqrt_cov1,torch.mul(diag_g1.variance,sqrt_cov1)))

    w2= delta_2_norm+torch.sum(mega_mat,dim=-1)
    return w2

if __name__ =='__main__':
   from torch.distributions import independent
   from torch.distributions import multivariate_normal,normal
   loc = torch.rand(3)
   scale= 2+torch.rand(3)
   xx= multivariate_normal.MultivariateNormal(loc,torch.eye(3)*scale)
   xy = multivariate_normal.MultivariateNormal(torch.zeros(3),torch.eye(3) )
   print (loc)
   print (scale)
   d1 = independent.Independent(normal.Normal(loc,scale),1)
   xz=multivariate_normal.MultivariateNormal(loc,torch.eye(3)*torch.pow(scale,2) )
   loc = torch.rand(3)
   scale = 2 + torch.rand(3)
   print(loc)
   print(scale)
   d2 = independent.Independent(normal.Normal(loc, scale), 1)
   d3 = independent.Independent(normal.Normal(torch.zeros(3),torch.ones(3)), 1)

   print (dist_W2_diag_1(d1, d2))
   print(dist_W2_diag_1(d1, d3))
   print ("tt ",dist_W2_indepen_mv(d1, xy))
   print("dd ",dist_W2_mv_mv(xz, xy))
   print(dist_W2_new_diag_standard(d1))


