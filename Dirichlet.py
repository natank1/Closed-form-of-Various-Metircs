import torch


# This function measure KL divgence for beta and Dirichlet distirbutuin
#  alpha abd beta are dirchlet distirbutions (in case of Beta dist simly take alpha and beta of size two each)
def Dirich_KL( alpha, beta  ):
    alpha0 =torch.sum(alpha ,dim=-1)
    beta0 = torch.sum(beta, dim=-1)
    alpha_term= torch.lgamma(alpha0)- torch.sum(torch.lgamma(alpha),dim=-1)
    beta_term=  torch.lgamma(beta0)- torch.sum(torch.lgamma(beta),dim=-1)
    overall_tem =alpha_term-beta_term
    del_ab = alpha-beta
    diag_a= torch.digamma(alpha)
    diag_a0 = torch.digamma(alpha0)
    del_diag = diag_a-diag_a0
    kl_metric = overall_tem +torch.sum(torch.mul(del_ab,del_diag),dim=-1)

    return kl_metric


def Dirich_KL_new( dir0,dir1 ):
    alpha0 =torch.sum(dir0.concentration ,dim=-1)
    beta0 = torch.sum(dir1.concentration, dim=-1)
    di0_term =torch.lgamma(alpha0) -torch.sum(torch.lgamma(dir0.concentration),dim=-1)
    dir1_term = torch.lgamma(beta0) - torch.sum(torch.lgamma(dir1.concentration), dim=-1)
    partial_term=di0_term-dir1_term

    tot_diggam = torch.unsqueeze(torch.digamma(alpha0),dim=-1)
    del_digamma =torch.digamma(dir0.concentration)-tot_diggam

    del_cocnetration =dir0.concentration- dir1.concentration
    term_4= torch.sum(torch.mul(del_digamma,del_cocnetration),dim=-1)

    kl_metric = term_4+partial_term

    return kl_metric
if __name__ =='__main__':
    a=0
    from torch.distributions.dirichlet import Dirichlet
    import torch.distributions.kl as kl0
    a=torch.tensor([[1.,1],[3,3.],[3,0.5],[0.5,3]])
    d= Dirichlet(a)
    b=torch.tensor([[3,3.],[1.,1],[0.5,3],[3,0.5]])
    d1 = Dirichlet(b)
    print (kl0._kl_dirichlet_dirichlet( d,d1))
    print( Dirich_KL_new(d,d1))
