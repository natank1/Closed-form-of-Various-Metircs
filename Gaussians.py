import torch


def kl_univ_gauss(mean1, sig1, mean2, sig2):
    #sig is stnafard  dev no varaince !
    kl_div=torch.log(sig2/sig1)+(torch.pow(sig1,2)+torch.pow(mean1-mean2,2))/(2*torch.pow(sig2,2)) -0.5
    return kl_div


def kl_mult_gauss(mean1, cov1, mean2, cov2,dimension):

    cov2_det=torch.det(cov2)
    cov1_det = torch.det(cov1)
    log_ratio = torch.log(cov2_det / cov1_det)
    inverse_2 =torch.inverse(cov2)
    tr_prod =torch.trace(torch.mm(inverse_2,cov1))
    delta_mean= mean1-mean2
    sq_prod= torch.matmul(delta_mean,torch.matmul(inverse_2,delta_mean))
    kl_div=0.5*(log_ratio-dimension+sq_prod +tr_prod)
    return kl_div

def kl_mult_gauss_diag(mean1, cov1, mean2, cov2,dimension,index=0):
    det_1= torch.prod(cov1,dim=index)
    det_2= torch.prod(cov2,dim=index)
    log_ratio = torch.log(det_2 / det_1)

    recip_2 = torch.reciprocal(cov2)

    delta_mean = mean1 - mean2


    mat_prod= torch.sum(torch.mul(delta_mean,torch.mul(recip_2,delta_mean)),dim=index)

    trace_like =torch.sum(torch.mul(recip_2,cov1))
    kl_div =0.5*(log_ratio-dimension+mat_prod+trace_like)
    return kl_div


def kl_mult_gauss_standard(mean1, cov1,dimension):
    # dimension = mean1.shape[index_loc]
    cov1_det = torch.det(cov1)
    log_cov = -torch.log(cov1_det)
    tr_cov =torch.trace(cov1)
    norm_mu = mean1.matmul(mean1)
    kl_div= 0.5 * (log_cov - dimension+ tr_cov + norm_mu)
    return kl_div


def kl_diag_standartd(mean1, cov1,dimension,index=0):
    cov1_det = torch.prod(cov1, dim=index)
    log_cov = -torch.log(cov1_det)
    tr_cov =torch.sum(cov1,dim=index)
    norm_mu = mean1.matmul(mean1)
    kl_div= 0.5 * (log_cov - dimension+ tr_cov + norm_mu)
    return kl_div


if __name__=='__main__':
    z= torch.tensor([1,2,3.,4.])
    z1 = torch.tensor([1, 2, 3., 4.])
    z2= torch.tensor([1, 2, 3., 4.])
    print (torch.mul(z,torch.mul(z1,z2)))
    print (torch.prod(z))
    print (torch.reciprocal(z))
    x= 2+torch.rand(size=(4,4))
    y =  torch.rand(4)
    y2 = torch.rand(4)

    t=torch.tensor([1,2,3,4.])
    t0 =torch.eye(4)*t
    u = torch.tensor([2.1, 3.1, 0.8, 2.7])
    u0 = torch.eye(4)
    print ("new values")
    print (kl_mult_gauss(y , t0, torch.zeros(4), u0,4))
    print(kl_mult_gauss_diag(y , t, torch.zeros(4), torch.ones(4),4, index=0))


    print ( kl_mult_gauss_standard(y, t0, 4))
    print (kl_diag_standartd(y, t,4,index=0))

    exit(12)
    x = torch.t(x).matmul(x)

    x2=5.2+torch.rand(size=(4,4))
    y2= 3+torch.rand(4)
    x2=torch.t(x2).matmul(x2)
    aa=kl_mult_gauss(y, x, y, x, 4)
    aa = kl_mult_gauss(y, x, y, x, 4)

    bb=kl_mult_gauss_standard(y, x, 4)
    bb = kl_mult_gauss_standard(y, x, 4)

    print ("aa=",aa)
    print("bb=",bb)
    print(kl_mult_gauss(y, x, torch.zeros(4),torch.eye(4),  4))

    exit(44)
    m1=torch.tensor([2.0,40])
    s1 = torch.tensor([3.0,5.0])
    m2 = torch.tensor([5.0, 40])
    s2 = torch.tensor([3.0, 5.0])

    print (torch.pow(m1-m2,2),torch.log(s2/s1))
    print (kl_univ_gauss(m1,s1,m2,s2))