import torch

# This is a KL divergence between two Gamm distirubtions
# In this function alpha0 and beta0 represent one Gamma dist and alpha1 and beta1 is another
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
def kl_gamma_new(gamma0,gamma1):
  diaggama0 =torch.digamma(gamma0.concentration)
  del_alpha =gamma0.concentration-gamma1.concentration
  term_digamma = torch.mul(del_alpha, diaggama0)
  del_logg= torch.lgamma(gamma1.concentration)-torch.lgamma(gamma0.concentration)


  term3= (torch.log(gamma0.rate)- torch.log(gamma1.rate))*gamma1.concentration
  # tot_alpha =tot_alpha +torch.mul(alpha1,logb0-logb1)
  beta_recip =torch.reciprocal(gamma0.rate)
  beta_ratio =torch.mul(gamma0.concentration,torch.mul(gamma1.rate-gamma0.rate,beta_recip))
  score =beta_ratio +term3+ term_digamma+del_logg
  return score
# In this function we work similarly to the previous but usin k and theta

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
     from torch.distributions.gamma import Gamma
     from torch.distributions.dirichlet import Dirichlet

     import torch.distributions.kl as kl0
     a=0
     g =Gamma(torch.tensor([2.0]), torch.tensor([1.0]))
     g1 = Gamma(torch.tensor([7.0]), torch.tensor([2.0]))
     x= torch.tensor([1.,2.0])
     x=torch.unsqueeze(x,dim=0)
     xt= x.repeat(10,1)
     gt= Gamma(xt[:,0],xt[:,1])
     print (xt.shape)
     print ("a", kl0._kl_gamma_gamma(g,g1))
     print ("b ", kl_gamma_new(g,g1))
     x = torch.tensor([2., 5.0])
     x = torch.unsqueeze(x, dim=0)
     xt = x.repeat(10, 1)
     gt1 = Gamma(xt[:, 0], xt[:, 1])
     print("a", kl0._kl_gamma_gamma(gt, gt1))
     print("b ", kl_gamma_new(gt, gt1))
     x0 =2+torch.rand(size=(10,2))
     y0 = 5 + torch.rand(size=(10, 2))
     gt0 = Gamma(x0[:, 0], x0[:, 1])
     gt1 = Gamma(y0[:, 0], y0[:, 1])
     print("aaa", kl0._kl_gamma_gamma(gt, gt1))
     print("bbbb ", kl_gamma_new(gt, gt1))
     print ("Dirichletirichlet")
     z = 1 + torch.rand(size=(10, 3))
     z1 = 4. + torch.rand(size=(10, 3))

     d0 = Dirichlet(torch.tensor([1.,1.]))
     d1 = Dirichlet(torch.tensor([3,3.]))

     print("aa1a", kl0._kl_dirichlet_dirichlet(d1, d0))
     print("1bbb", Dirich_KL_new(d1, d0))

     z=1+torch.rand(size=(10,3))
     z1 = 4.+torch.rand(size=(10, 3))
     print (z.shape)
     d0 = Dirichlet(z)
     d1 = Dirichlet(z1)



     print("3aaa", kl0._kl_dirichlet_dirichlet(d0, d1))
     print("3bbb", Dirich_KL_new(d0, d1))


