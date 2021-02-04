import torch
import scipy.linalg as lin_alg
import numpy as np
def dist_W2_torch(mean1, cov1, mean2, cov2):
    delta_mean = mean1-mean2
    delta_2_norm= delta_mean.matmul(delta_mean)
    sqrt_cov1= lin_alg.fractional_matrix_power( cov1, 0.5)
    mega_mat=  2*torch.tensor(lin_alg.fractional_matrix_power( sqrt_cov1*cov2.detach().numpy()*sqrt_cov1 , 0.5))
    w2=delta_2_norm+torch.trace(cov1+cov2-mega_mat)
    return w2

def dist_W2_torch_diag(mean1, cov1, mean2, cov2,index):
    delta_mean = mean1-mean2
    delta_2_norm= delta_mean.matmul(delta_mean)

    sqrt_cov1= torch.sqrt( cov1)
    mega_mat=  cov1+cov2-2*torch.sqrt(torch.mul(sqrt_cov1,torch.mul(cov2,sqrt_cov1)))
    w2= delta_2_norm+torch.sum(mega_mat,dim=index)
    return w2



def dist_W2_torch_standard(mean1, cov1, dist_dim):
    # Inouts are mean and varaice (not std! ,varaince)
    # lat_dim = mean1.shape[sample_index]
    norm_mu=mean1.matmul(mean1)

    trace_of_mega= 2*np.trace( lin_alg.fractional_matrix_power(cov1,0.5))

    w2_score = norm_mu  +torch.trace(cov1)+dist_dim -torch.tensor(trace_of_mega)

    return w2_score


def dist_W2_diag_standard(mean1, cov1, dimension,index):
    norm_mu = mean1.matmul(mean1)
    mega_trace=2* torch.sum(torch.sqrt(cov1),dim=index)
    tr1 =torch.sum(cov1,dim=0)
    w2= norm_mu+tr1+dimension -mega_trace
    return w2


if __name__=='__main__':
    x= 2+torch.rand(size=(4,4))
    y =  torch.rand(4)
    x = torch.t(x).matmul(x)
    x2 = 5.2 + torch.rand(size=(4, 4))
    y2 = 3 + torch.rand(4)
    x2 = torch.t(x2).matmul(x2)
    t=torch.tensor([1.4,2,2.1,3])
    z=torch.eye(4)*t
    u = torch.tensor([2.1, 0.9, 4.1, 0.3])
    z0 = torch.eye(4) * u
    print ("old ")
    print(dist_W2_torch(y,z,y2,z0))
    print(dist_W2_torch_diag(y, t, y2, u, 0))

    # print (dist_W2_torch(y, x, y2, x2))
    print ("on your mark")
    print(dist_W2_torch(y, z, torch.zeros(4),torch.eye(4) ))

    print (dist_W2_torch_standard(y,z,4))
    print(dist_W2_torch_diag(y, t, torch.zeros(4), torch.ones(4), 0))
    print(dist_W2_diag_standard(y, t, 4, 0))
    exit(444)

    cc =lin_alg.fractional_matrix_power(x,0.5)
    cd = lin_alg.fractional_matrix_power(x2, 0.5)
    v=cc*x2.detach().numpy()*cc
    cd = lin_alg.fractional_matrix_power(v, 0.5)
    print(np.trace(cd))

    print (torch.trace(torch.tensor(cd)))

    exit(4444)
    c2=torch.tensor(cc)
    print (c2.t().matmul(c2))


    exit(333)

    x2=5.2+torch.rand(size=(4,4))
    y2= 3+torch.rand(4)
    x2=torch.t(x2).matmul(x2)
    # aa=dist_W2_torch(y, x, y, x, 0)
    # aa = dist_W2_torch(y, x, y, x, 0)
