import torch



def Dirich_KL( alpha, beta, index ):
    alpha0 =torch.sum(alpha ,dim=index)
    beta0 = torch.sum(beta, dim=index)
    alpha_term= torch.lgamma(alpha0)- torch.sum(torch.lgamma(alpha),dim=index)
    beta_term=  torch.lgamma(beta0)- torch.sum(torch.lgamma(beta),dim=index)
    overall_tem =alpha_term-beta_term
    del_ab = alpha-beta
    diag_a= torch.digamma(alpha)
    diag_a0 = torch.digamma(alpha0)
    del_diag = diag_a-diag_a0
    kl_metric = overall_tem +torch.sum(torch.mul(del_ab,del_diag),dim=index)

    return kl_metric

if __name__ =='__main__':
    a=0
    a=torch.tensor([3.,0.5])
    b=torch.tensor([0.5,3.])
     
    print (Dirich_KL( b, a, 0 ))