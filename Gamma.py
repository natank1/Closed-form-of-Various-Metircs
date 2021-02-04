import torch
def kl_gamma_ab(alpha0,beta0,alpha1,beta1,index=0):
  digaga_alpha =torch.digamma(alpha0)
  del_alpha =alpha0-alpha1
  tot_alpha =torch.mul(del_alpha,digaga_alpha)
  tot_alpha=  tot_alpha -torch.lgamma(alpha0)+torch.lgamma(alpha1)
  logb0 = torch.log(beta0)
  logb1 = torch.log(beta1)
  tot_alpha =tot_alpha +torch.mul(alpha1,logb0-logb1)
  beta_recip =torch.reciprocal(beta0)
  beta_ratio =torch.mul(alpha1,torch.mul(beta1-beta0,beta_recip))
  tot_alpha=tot_alpha+beta_ratio
  return tot_alpha

def kl_gamma_ktheta(k0,th0,k1,th1,index=0):
    digaga_alpha = torch.digamma(k0)
    del_alpha = k0 - k1
    tot_alpha = torch.mul(del_alpha, digaga_alpha)
    tot_alpha = tot_alpha - torch.lgamma(k0) + torch.lgamma(k1)
    logb0 = torch.log(th0)
    logb1 = torch.log(th1)
    tot_alpha =tot_alpha +torch.mul(k1,logb1-logb0)
    beta_recip = torch.reciprocal(th1)
    beta_ratio = torch.mul(k0, torch.mul(th0 - th1, beta_recip))
    tot_alpha = tot_alpha + beta_ratio
    return tot_alpha


if __name__ =='__main__':
     a=0
     a = torch.tensor([3.])
     a1= torch.tensor([0.5])
     b = torch.tensor([0.5])
     b1=torch.tensor([3.])

     print(kl_gamma_ktheta(a,b,a1,b1, 0))
     print(kl_gamma_ab(a,b,a1,b1,index=0))