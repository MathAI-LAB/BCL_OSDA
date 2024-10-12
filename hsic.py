import torch
import numpy as np
from torch.autograd import Variable, grad

def sigma_estimation(X, Y):
    """ sigma from median distance
    """
    D = distmat(torch.cat([X,Y]))
    D = D.detach().cpu().numpy()
    #med =  np.mean(D)
    Itri = np.tril_indices(D.shape[0], -1)
    Tri = D[Itri]
    med = np.median(Tri)
    if med <= 0:
        med=np.mean(Tri)
    if med<1E-2:
        med=1E-2
    return med

def distmat(X):
    """ distance matrix
    """
    r = torch.sum(X*X, 1)
    r = r.view([-1, 1])
    a = torch.mm(X, torch.transpose(X,0,1))
    D = r.expand_as(a) - 2*a +  torch.transpose(r,0,1).expand_as(a)
    D = torch.abs(D)
    return D


def K_matrix(X,a):
    # input.size = (fea_dim*sample_size)                
    xt_x = X.pow(2).sum(0).unsqueeze(0) #torch.Size([1, 1191])
    a2 = xt_x.t().repeat(1,xt_x.shape[1])
    b2 = xt_x.repeat(xt_x.shape[1],1)
    ab = torch.mm(X.t(),X)
    dis_x = a2 + b2 -2*ab
    sigma = dis_x.mean() # = 2*sigma.pow(2)           
    K_x = torch.exp(-dis_x/sigma)
    H = torch.diag(a) - torch.mm(a,a.T)
    Kxc = torch.mm(K_x,H)
    
    return Kxc

def K_matrix0(X):
    # input.size = (fea_dim*sample_size)                
    m = int(X.size()[0])
    dis_x = distmat(X)
    sigma = dis_x.mean() # = 2*sigma.pow(2)           
    K_x = torch.exp(-dis_x/sigma).cuda()
    H = (torch.eye(m) - (1./m) * torch.ones([m,m])).cuda()
    Kxc = torch.mm(K_x,H)
    return Kxc

def kernelmat(X, sigma):
    """ kernel matrix baker
    """
    m = int(X.size()[0])
    dim = int(X.size()[1]) * 1.0
    H = torch.eye(m) - (1./m) * torch.ones([m,m])
    Dxx = distmat(X)
    
    if sigma:
        variance = 2.*sigma*sigma*X.size()[1]            
        Kx = torch.exp( -Dxx / variance).type(torch.FloatTensor)   # kernel matrices        
        # print(sigma, torch.mean(Kx), torch.max(Kx), torch.min(Kx))
    else:
        try:
            sx = sigma_estimation(X,X)
            Kx = torch.exp( -Dxx / (2.*sx*sx)).type(torch.FloatTensor)
        except RuntimeError as e:
            raise RuntimeError("Unstable sigma {} with maximum/minimum input ({},{})".format(
                sx, torch.max(X), torch.min(X)))
    Kxc = torch.mm(Kx,H)


    return Kxc

def kernelmat0(X,a, sigma):
    """ kernel matrix baker
    """
    m = int(X.size()[0])
    dim = int(X.size()[1]) * 1.0
    H = torch.diag(a) - torch.mm(a,a.T)
    Dxx = distmat(X)
    
    if sigma:
        variance = 2.*sigma*sigma*X.size()[1]            
        Kx = torch.exp( -Dxx / variance).type(torch.FloatTensor)   # kernel matrices        
        # print(sigma, torch.mean(Kx), torch.max(Kx), torch.min(Kx))
    else:
        try:
            sx = sigma_estimation(X,X)
            Kx = torch.exp( -Dxx / (2.*sx*sx)).type(torch.FloatTensor)
        except RuntimeError as e:
            raise RuntimeError("Unstable sigma {} with maximum/minimum input ({},{})".format(
                sx, torch.max(X), torch.min(X)))
    Kx = Kx.cuda()
    Kxc = torch.mm(Kx,H)


    return Kxc

def R_operator(K, epsilon ):
    n_samples = K.shape[0]
    H = (torch.eye(n_samples) - torch.ones((n_samples,n_samples))/n_samples).cuda()   
    G = torch.mm(torch.mm(H,K),H)
    R = torch.mm(G, torch.inverse(G+n_samples*epsilon*torch.eye(n_samples).cuda()) )
    return R

def c_hsic(X,Y,Z,epsilon=5e-4):
    X_all = torch.cat((X,Z),1) # torch.size([d + c, ns + nt])
    Y_all = torch.cat((Y,Z),1) # torch.Size([2 + c, ns + nt])
    
    K_x = kernelmat(X_all,sigma=None).cuda()
    K_y = kernelmat(Y_all,sigma=None).cuda()
    K_z = kernelmat(Z,sigma=None).cuda()
    
    
    R_x = R_operator(K_x,epsilon)
    R_y = R_operator(K_y,epsilon)
    R_z = R_operator(K_z,epsilon)
    
    # R_total = torch.mm(R_y,R_x) -2*torch.mm(torch.mm(R_y,R_x), R_z) + torch.mm(torch.mm(torch.mm(R_y,R_z),R_x),R_z)
    R_total = torch.mm(R_y,R_x) -torch.mm(torch.mm(R_y,R_x), R_z) - torch.mm(torch.mm(R_y,R_z), R_x) + torch.mm(torch.mm(torch.mm(R_y,R_z),R_x),R_z)
    COND_loss = torch.trace(R_total)
    return COND_loss
        

def distcorr(X, sigma=1.0):
    X = distmat(X)
    X = torch.exp( -X / (2.*sigma*sigma))
    return torch.mean(X)

def compute_kernel(x, y):
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    x = x.unsqueeze(1) # (x_size, 1, dim)
    y = y.unsqueeze(0) # (1, y_size, dim)
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    kernel_input = (tiled_x - tiled_y).pow(2).mean(2)/float(dim)
    return torch.exp(-kernel_input) # (x_size, y_size)

def mmd(x, y, sigma=None, use_cuda=True, to_numpy=False):
    m = int(x.size()[0])
    H = torch.eye(m) - (1./m) * torch.ones([m,m])
    # H = Variable(H)
    Dxx = distmat(x)
    Dyy = distmat(y)

    if sigma:
        Kx  = torch.exp( -Dxx / (2.*sigma*sigma))   # kernel matrices
        Ky  = torch.exp( -Dyy / (2.*sigma*sigma))
        sxy = sigma
    else:
        sx = sigma_estimation(x,x)
        sy = sigma_estimation(y,y)
        sxy = sigma_estimation(x,y)
        Kx = torch.exp( -Dxx / (2.*sx*sx))
        Ky = torch.exp( -Dyy / (2.*sy*sy))
    # Kxc = torch.mm(Kx,H)            # centered kernel matrices
    # Kyc = torch.mm(Ky,H)
    Dxy = distmat(torch.cat([x,y]))
    Dxy = Dxy[:x.size()[0], x.size()[0]:]
    Kxy = torch.exp( -Dxy / (1.*sxy*sxy))

    mmdval = torch.mean(Kx) + torch.mean(Ky) - 2*torch.mean(Kxy)

    return mmdval

def mmd_pxpy_pxy(x,y,sigma=None,use_cuda=True, to_numpy=False):
    """
    """
    if use_cuda:
        x = x.cuda()
        y = y.cuda()
    m = int(x.size()[0])

    Dxx = distmat(x)
    Dyy = distmat(y)
    if sigma:
        Kx  = torch.exp( -Dxx / (2.*sigma*sigma))   # kernel matrices
        Ky  = torch.exp( -Dyy / (2.*sigma*sigma))
    else:
        sx = sigma_estimation(x,x)
        sy = sigma_estimation(y,y)
        sxy = sigma_estimation(x,y)
        Kx = torch.exp( -Dxx / (2.*sx*sx))
        Ky = torch.exp( -Dyy / (2.*sy*sy))
    A = torch.mean(Kx*Ky)
    B = torch.mean(torch.mean(Kx,dim=0)*torch.mean(Ky, dim=0))
    C = torch.mean(Kx)*torch.mean(Ky)
    mmd_pxpy_pxy_val = A - 2*B + C 
    return mmd_pxpy_pxy_val

def hsic_regular(x, y, sigma=None, use_cuda=True, to_numpy=False):
    """
    """
    Kxc = K_matrix0(x)
    Kyc = K_matrix0(y)
    KtK = torch.mul(Kxc, Kyc.t())
    Pxy = torch.mean(KtK)
    return Pxy


def hsic_regular0(x, y, a, sigma=None, use_cuda=True, to_numpy=False):
    """
    """
    Kxc = K_matrix(x.T,a)
    Kyc = K_matrix(y.T,a)
    KtK = torch.mul(Kxc, Kyc.t())
    Pxy = torch.mean(KtK)
    return Pxy


def hsic_normalized(x, y, sigma=None, use_cuda=True, to_numpy=True):
    """
    """
    m = int(x.size()[0])
    Pxy = hsic_regular(x, y, sigma, use_cuda)
    Px = torch.sqrt(hsic_regular(x, x, sigma, use_cuda))
    Py = torch.sqrt(hsic_regular(y, y, sigma, use_cuda))
    thehsic = Pxy/(Px*Py)
    return thehsic

def hsic_normalized_cca(x, y, sigma, use_cuda=True, to_numpy=True):
    """
    """
    m = int(x.size()[0])
    Kxc = kernelmat(x, sigma=sigma)
    Kyc = kernelmat(y, sigma=sigma)

    epsilon = 1E-5
    K_I = torch.eye(m)
    Kxc_i = torch.inverse(Kxc + epsilon*m*K_I)
    Kyc_i = torch.inverse(Kyc + epsilon*m*K_I)
    Rx = (Kxc.mm(Kxc_i))
    Ry = (Kyc.mm(Kyc_i))
    Pxy = torch.sum(torch.mul(Rx, Ry.t()))

    return Pxy


