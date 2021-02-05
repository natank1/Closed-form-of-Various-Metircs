import torch

#  This file contains set of functions for calcualting KL divergence for different scenarios  of Gaussians

#Univaraite case we simply receive two means and two standard deviations
def kl_univ_gauss(mean1, sig1, mean2, sig2):
    #sig is stnafard  dev no varaince !
    kl_div=torch.log(sig2/sig1)+(torch.pow(sig1,2)+torch.pow(mean1-mean2,2))/(2*torch.pow(sig2,2)) -0.5
    return kl_div

# The batch case (nahdling batch broadcasting of matrices)
def kl_mult_gauss_batch(mean1, cov1, mean2, cov2,dimension):

    # cov2_det=torch.det(cov2)
    # cov1_det = torch.det(cov1)
    # log_ratio = torch.log(cov2_det / cov1_det)
    log_ratio = torch.logdet(cov2) - torch.logdet(cov1)
    inverse_2 =torch.inverse(cov2)

    tmo_mat =torch.matmul(inverse_2,cov1)
    tr_prod= torch.diagonal(tmo_mat, dim1=-2, dim2=-1).sum(-1)

    delta_mean= torch.unsqueeze(mean1-mean2,dim=-1 )
    aa= torch.matmul(inverse_2,delta_mean)
    sq_prod= torch.squeeze(torch.squeeze(torch.matmul(torch.transpose(delta_mean,dim0=-2,dim1=-1),aa),dim=-1),dim=-1)

    kl_div=0.5*(log_ratio-dimension+sq_prod +tr_prod)
    return kl_div


# The multivariate case (no batch) : mean1 adn mean2 are vectors (for the means ) and cov1 and cov2 are covraint matrix
def kl_mult_gauss(mean1, cov1, mean2, cov2,dimension):



    log_ratio = torch.logdet(cov2) - torch.logdet(cov1)
    inverse_2 =torch.inverse(cov2)
    tr_prod =torch.trace(torch.mm(inverse_2,cov1))

    delta_mean= mean1-mean2
    sq_prod= torch.matmul(delta_mean,torch.matmul(inverse_2,delta_mean))
    kl_div=0.5*(log_ratio-dimension+sq_prod +tr_prod)
    return kl_div

# Here we assume that the covaraince matrices are diagonal hence held as vectors
def kl_mult_gauss_diag(mean1, cov1, mean2, cov2,dimension,index=0):
    log_ratio= torch.sum(torch.log(cov2),dim=index) - torch.sum(torch.log(cov1),dim=index)
    recip_2 = torch.reciprocal(cov2)
    delta_mean = mean1 - mean2
    mat_prod= torch.sum(torch.mul(delta_mean,torch.mul(recip_2,delta_mean)),dim=index)
    trace_like =torch.sum(torch.mul(recip_2,cov1))
    kl_div =0.5*(log_ratio-dimension+mat_prod+trace_like)
    return kl_div

# The notion standrrd assumes that we compare the matrix to the standrard Gaussian thus
# we have obly a single gaussian
def kl_mult_gauss_standard(mean1, cov1,dimension,index):


    log_cov =torch.logdet(cov1)
    tr_cov =torch.trace(cov1)
    norm_mu = torch.pow(torch.norm(mean1,dim=index),2)
    kl_div= 0.5 * (log_cov - dimension+ tr_cov + norm_mu)
    return kl_div


def kl_diag_standartd(mean1, cov1,dimension,index=0):

    log_cov = -torch.sum(torch.log(cov1),dim=index)
    tr_cov =torch.sum(cov1,dim=index)
    norm_mu = torch.pow(torch.norm(mean1,dim=index),2)
    kl_div= 0.5 * (log_cov - dimension+ tr_cov + norm_mu)
    return kl_div

