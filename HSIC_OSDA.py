# -*- coding: utf-8 -*-

##################
import torch
torch.cuda.set_device(0)
##################

from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import warnings
import numpy as np
from datetime import datetime
import scipy.io as io 
import os
from hsic import hsic_regular,hsic_regular0,c_hsic
import random
from sklearn.cluster import KMeans
warnings.filterwarnings("ignore")
# Training settings
#torch.manual_seed(1) 

#====== Hyper-Parameter ======
# devices = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
devices = torch.device('cuda',0)
epochs = 600
Exptimes = 5
n_classes = 26
TarEnt_step = 50

save_path = '/home/legion-1/LabData/Hzx/model/'
lambda1 = [0.8e-2]
lambda2 = [2.8e-1]
#
# #====== ImageCLEF ======
#dset = 'ImageCLEF'
#source_domain_set = ('I','P','I','C','C','P')
#target_domain_set = ('P','I','C','I','P','C')
#save_name = ('ItoP','PtoI','ItoC','CtoI','CtoP','PtoC')
#partial_cls_idx = [0,1,2,3,4,5]

#====== OfficeHome ======
# dset = 'OfficeHome'
# source_domain_set = ('Art','Art','Art','Clipart','Clipart','Clipart','Product','Product','Product','Real_World','Real_World','Real_World')
# target_domain_set = ('Clipart','Product','Real_World','Art','Product','Real_World','Art','Clipart','Real_World','Art','Clipart','Product')
# save_name = ('ARtoCL','ARtoPR','ARtoRW','CLtoAR','CLtoPR','CLtoRW','PRtoAR','PRtoCL','PRtoRW','RWtoAR','RWtoPR','RWtoRW')
# partial_cls_idx = list(range(25))

#====== Office31 TIDOT ====
# dset = 'Office31TIDOT'
# n_classes = 31
# source_domain_set = ('amazon','amazon','dslr','dslr','webcam','webcam')
# target_domain_set = ('dslr','webcam','amazon','webcam','amazon','dslr')
# save_name = ('AtoD','AtoW','DtoA','DtoW','WtoA','WtoD')
# partial_cls_idx = [0,1,5,10,11,12,15,16,17,22]

#====== Office31 ======
dset = 'OfficeHome'
n_classes = 26
source_domain_set = ('Art')
#'webcam','dslr','dslr','amazon','amazon','dslr'
target_domain_set = ('Product')
#'dslr','amazon','webcam','dslr','webcam','amazon'
save_name = ('AtoP')
#,'WtoA','WtoD','DtoA','DtoW'
source_cls_idx = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
target_cls_idx = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64]



##################################
# Load data and create Data loaders
##################################
def get_mat_path(dset,domain):
    data_dir = '/home/legion-1/LabData/Hzx'
    if dset == 'ImageCLEF':
        dset_dir = 'ImageCLEF-DA'
        if domain == 'I':
            file_name = 'imageclef-i-resnet50-noft.mat'
        elif domain == 'P':
            file_name = 'imageclef-p-resnet50-noft.mat'
        elif domain == 'C':
            file_name = 'imageclef-c-resnet50-noft.mat'
    elif dset == 'OfficeHome':
        dset_dir = 'OfficeHome_Resnet50'
        if domain == 'Art':
            file_name = 'OfficeHome-Art-resnet50-noft.mat'
        elif domain == 'Clipart':
            file_name = 'OfficeHome-Clipart-resnet50-noft.mat'
        elif domain == 'Product':
            file_name = 'OfficeHome-Product-resnet50-noft.mat'
        elif domain == 'Real_World':
            file_name = 'OfficeHome-RealWorld-resnet50-noft.mat'
    elif dset == 'Office10':
        dset_dir = 'Office-Caltech'
        if domain == 'A':
            file_name = 'amazon_decaf.mat'
        elif domain == 'C':
            file_name = 'caltech_decaf.mat'
        elif domain == 'D':
            file_name = 'dslr_decaf.mat'
        elif domain == 'W':
            file_name = 'webcam_decaf.mat'
    elif dset == 'Office31':
        dset_dir = 'Office31_Resnet50'
        if domain == 'amazon':
            file_name = 'office-A-resnet50-noft.mat'
        elif domain == 'webcam':
            file_name = 'office-W-resnet50-noft.mat'
        elif domain == 'dslr':
            file_name = 'office-D-resnet50-noft.mat'
    elif dset == 'Morden31':
        dset_dir = 'Modern-Office-31'
        if domain == 'amazon':
            file_name = 'Modern-Office-31-amazon-resnet50-noft.mat'
        elif domain == 'webcam':
            file_name = 'Modern-Office-31-webcam-resnet50-noft.mat'
        elif domain == 'synthetic':
            file_name = 'Modern-Office-31-synthetic-resnet50-noft.mat'
    elif dset == 'Office31TIDOT':
            dset_dir = 'Office31_TIDOT_IJCAI21'
            if domain == 'amazon':
                file_name = 'office-A-resnet50-noft-tidot.mat'
            elif domain == 'webcam':
                file_name = 'office-W-resnet50-noft-tidot.mat'
            elif domain == 'dslr':
                file_name = 'office-D-resnet50-noft-tidot.mat'
    elif dset == 'Adaptiope':
        dset_dir = 'Adaptiope'
        if domain == 'Product':
            file_name = 'Adaptiope-product_images-resnet50-noft.mat'
        elif domain == 'Real_Life':
            file_name = 'Adaptiope-real_life-resnet50-noft.mat'
        elif domain == 'Synthetic':
            file_name = 'Adaptiope-synthetic-resnet50-noft.mat'
            
    path = os.path.join(data_dir,dset_dir,file_name)
    return path
    



class ImageSet_dataset(Dataset):
    def __init__(self, mat_path, Office10 = False,partial_cls_idx=False):
        # Exp_Type must be 'Train', 'Test' or 'Valid'
        data = io.loadmat(mat_path)
        if Office10 is True:
            img = torch.from_numpy(data['feas']).float()
            label = torch.from_numpy((data['labels']-1).squeeze(1)).long()
        else:
            img = torch.from_numpy(data['resnet50_features'])
            img = img.view([img.shape[0],-1])
            label = torch.from_numpy(data['labels'].squeeze(0))
        
        if partial_cls_idx:
            sam_idx = (torch.zeros(len(label))==1)
            for cls_idx in partial_cls_idx:
                sam_idx += (label==cls_idx)
            img = img[sam_idx,:]
            label = label[sam_idx]
        
        self.img = img
        self.label = label
#        self.label = label - 1
        del label, img
        
    def __getitem__(self, idx):
        Batch_data = self.img[idx,:]
        Batch_label = self.label[idx]
        return Batch_data, Batch_label
    
    def __len__(self):
        return len(self.label)
    
def Make_Loader(mat_path, Batch_size = None, Office10 = False, partial_cls_idx = False):
    data_set = ImageSet_dataset(mat_path, Office10, partial_cls_idx)
    if Batch_size is None:
        Batch_size = len(data_set)
    new_loader = DataLoader(data_set, Batch_size, shuffle=False)
    return new_loader

##################################
# Define Networks
##################################
class C(nn.Module):
    def __init__(self, input_dim, conv_dim_1, conv_dim_2):
        super(C, self).__init__()
        self.fc1 = nn.Sequential( 
              nn.Linear(input_dim, conv_dim_1), 
              nn.BatchNorm1d(conv_dim_1),
              nn.LeakyReLU(negative_slope=0.2, inplace=True), 
             ) 
        self.fc2 = nn.Sequential( 
              nn.Linear(conv_dim_1, conv_dim_2), 
              nn.BatchNorm1d(conv_dim_2),
              nn.Tanh(),
             ) 
        self.fc3 = nn.Linear(conv_dim_2, n_classes) 


    def forward(self, x):
        z_1 = self.fc1(x)
        z_2 = self.fc2(z_1)
        y = self.fc3(z_2)
        p = F.softmax(y)
        return z_1, z_2, p

     
def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.05)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
        elif classname.find('Linear') != -1:
            m.weight.data.normal_(0.0, 0.05)
            m.bias.data.fill_(0)         
########################         
def reset_grad():
    """Zeros the gradient buffers."""
    c.zero_grad()
    
def to_var(x):
    """Converts numpy to variable."""
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def to_data(x):
    """Converts variable to numpy."""
    if torch.cuda.is_available():
        x = x.cpu()
    return x.data.numpy()   
    

def classification_accuracy_train():
    with torch.no_grad():
        correct = 0

        iter_loader_s = iter(source_loader)
        X_s, lab_s = iter_loader_s.next()
        X_s, lab_s = to_var(X_s), to_var(lab_s).long().squeeze()
        _, _, output = c(X_s)           
        pred = output.data.max(1)[1]
        correct += pred.eq(lab_s.data).cpu().float()
        
        return correct.mean().item()    

def classification_accuracy():
    with torch.no_grad():
        correct = 0

        iter_loader_t = iter(target_loader)
        X_t, lab_t = iter_loader_t.next()
        X_t, lab_t = to_var(X_t), to_var(lab_t).long().squeeze()
        lab_t = torch.where(lab_t>25,25,lab_t)
        _, _, output = c(X_t)
        #print(output)
        pred = output.data.max(1)[1]
        #print(torch.sum(lab_t==4))
        #print(torch.sum((pred==lab_t)*(lab_t==4))/torch.sum(lab_t==4))
        #print(torch.sum((lab_t==4)*(pred==10)))
        acc_per_class = torch.zeros(n_classes)
        for i in range(n_classes):
            acc_per_class[i] = torch.sum((pred==lab_t)*(lab_t==i))/torch.sum(lab_t==i)
        print(acc_per_class)
    OS = torch.mean(acc_per_class[0:n_classes-1])
    Unk = acc_per_class[n_classes-1]
    correct = (2*OS*Unk/(OS+Unk)).cpu().float()
        
    return correct.item(), OS, Unk


def Entropy(x):
    epsilon = 1e-5
    entropy = -x*torch.log(x+epsilon)
    entropy = torch.sum(entropy,dim=1)
    return entropy






################################### Loss Func ##################################
def Pred_Entropy_loss(pred_t):
    num_sam = pred_t.shape[0]
    Entropy = -(pred_t.mul(pred_t.log()+1e-4)).sum()
    
    return Entropy/num_sam

def One_Hot(lab, num_cls):
    num_sam = lab.shape[0]
    
    lab_mat = torch.zeros([num_sam,num_cls]).cuda().scatter_(1, lab.unsqueeze(1), 1)
    
    return lab_mat

def Weighting_ERM_loss(pred, lab, num_cls, weight_ratio):
    # p: source density, q: target density
    # pred: [sample-size * class number] posterior prediction matrix of source data
    # lab: [sample-size] ground-truth label vector of source data
    # num_cls: scalar of class number
    # weight_ratio: [class number] class weight vector as q_y/p_y
    one_hot_lab = One_Hot(lab, num_cls)
    pred_score = (one_hot_lab.mul(pred)).sum(1)
    Cross_Entropy = -(pred_score+1e-6).log()
    pred_weight = weight_ratio[lab]
    
    weighting_ERM = (Cross_Entropy.mul(pred_weight)).mean()
    
    return weighting_ERM

def dist(x,y):
    xx = torch.sum(torch.mul(x,x),dim=1).reshape(x.shape[0],1).repeat(1,y.shape[0])
    yy = torch.sum(torch.mul(y,y),dim=1).reshape(y.shape[0],1).repeat(1,x.shape[0])
    xy = torch.mm(x,y.T)
    cost = (xx+yy.T)-2*xy
    return cost

def path_maker(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    print("create!")
    

def sim(M):
    #xx = torch.sum(torch.mul(M, M), dim=1).reshape(M.shape[0], 1)
    xy = torch.mm(M.T, M)
    xy=xy.T
    min_val,_=torch.min(xy,dim=0)
    max_val,_=torch.max(xy,dim=0)
    ans = (xy-min_val)/(max_val-min_val)
    ans =ans.T
    return ans

########################

# Downlaod Pretrained Model
Detail_result = ()
Max_ACC = np.zeros((Exptimes+1,len(save_name)))

base_path = '/home/legion-1/LabData/Hzx/result0'+'/' + dset + '/' + 'analyse'
path_maker(base_path)

for lambda2_i in range(len(lambda2)):
    lambda00 = lambda2[lambda2_i]
    for lambda1_i in range(len(lambda1)):
            lambda0 = lambda1[lambda1_i]
            for Domain_iter in range(1):
                seed = 1024
                torch.cuda.manual_seed(seed) # 为当前GPU设置随机种子
                torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU，为所有GPU设置随机种子
                np.random.seed(seed)  # Numpy module.
                random.seed(seed)  # Python random module.	
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
                os.environ['PYTHONHASHSEED'] = str(seed)
                source_domain = source_domain_set
                target_domain = target_domain_set
                source_path = get_mat_path(dset,source_domain)
                target_path = get_mat_path(dset,target_domain)
                # source_loader = Make_Loader(source_path,Office10=True)
                # target_loader = Make_Loader(target_path,Office10=True)
                source_loader = Make_Loader(source_path, partial_cls_idx=source_cls_idx)
                target_loader = Make_Loader(target_path, partial_cls_idx = target_cls_idx)
                
                Total_Result = np.zeros((8,epochs,Exptimes))
                # Random Experiment
                Exp_iter = 0
                while Exp_iter < Exptimes:
                    # Define the network architech
                    c = C(2048, 1024, 512)
                
                    c.cuda()
                    c.apply(weights_init)
                    # torch.save(c.state_dict(),save_path+f'base_{save_name[Domain_iter]}.pth')
                    # c.load_state_dict(torch.load(save_path+f'base_{save_name[Domain_iter]}.pth'))
                    
            
                            
                    lr=0.0003
                    beta1=0.9
                    beta2=0.999
                    
                    c_solver = optim.Adam([{'params':c.parameters()}], lr, [beta1, beta2], weight_decay=0.01)
                    criterionQ_dis = nn.NLLLoss().cuda() 
                                        
                    ####################
                    # Train procedure
                    ####################
                    result = np.zeros((8,epochs))
                    start_time = datetime.now()
                    Exp_start_time = datetime.now()
                    
                    print("************ %1s→%1s:  %1s Start Experiment %1s training ************"%(source_domain,target_domain,Exp_start_time,Exp_iter+1))  
                    for step in range(epochs):
                        epoch_time_start = datetime.now()
            
                        Current_loss = np.array([0])
                        Current_TarEnt_loss = np.array([0])  
                 
                        iter_loader_s = iter(source_loader)
                        iter_loader_t = iter(target_loader)
                        X_s, lab_s = iter_loader_s.next()
                        X_t, lab_t = iter_loader_t.next()
                        c.train()
                        
                        X_s, lab_s = Variable(X_s), Variable(lab_s)
                        X_s, lab_s = X_s.cuda(), lab_s.cuda()
                        X_t, lab_t = Variable(X_t),Variable(lab_t)
                        X_t, lab_t = X_t.cuda(), lab_t.cuda()
                        lab_t = torch.where(lab_t>25,25,lab_t)
                        lab_s0 = lab_s.cpu().numpy()
                            
                            # Init gradients
                        reset_grad()
                        h_s, h_s2, pred_s = c(X_s)
                        h_t, h_t2, pred_t = c(X_t)
            
            
                        n = lab_s.shape[0]
                        m = lab_t.shape[0]
            
            
                        
                        fs_norm = (torch.norm(h_s2, 2, dim=1).repeat(h_s2.shape[1], 1)).T
                        ft_norm = (torch.norm(h_t2, 2, dim=1).repeat(h_t2.shape[1], 1)).T
                        fs = h_s2 / fs_norm
                        ft = h_t2 / ft_norm
                        # fs = h_s2
                        # ft = h_t2
                        
                        
                        
                        
                        ys_one_hot = torch.zeros(n,n_classes).cuda()
                        ys_one_hot = ys_one_hot.scatter_(1,lab_s.reshape(n,1),1)
                        yt_one_hot = torch.zeros(m,n_classes).cuda()
                        
                        
                        entropy_t = Entropy(pred_t).reshape(1,-1).detach()
                        entropy_t = entropy_t.cpu().numpy().T
                        cluster = KMeans(n_clusters=3,random_state=0).fit(entropy_t)
                        entroid = cluster.cluster_centers_
                        index = np.argmax(entroid)
                        index_u = (cluster.labels_ == index)
                        index_u = torch.from_numpy(index_u)
                        index_u = torch.squeeze(index_u).cuda()
                        
            
            
                        pred_t_NG = pred_t.detach()
                        pse_t = pred_t_NG.data.max(1)[1]
                        yt_one_hot = yt_one_hot.scatter_(1,pse_t.reshape(m,1),1)
                        
                        
                       
                        uuu = (pse_t==25)

                        
                        # if step>=60:
                        #     index_u = (index_u&uuu)
                
                        h_k1 = ~index_u
                        I_c = 0;
                        for i in range(n_classes-1):
                            index = (pse_t==i)
                            index_t = (h_k1&index)
                            index_s = (lab_s0==i)
                            if ~(index_t.any()):
                                continue
                            m1 = fs[index_s,:].shape[0]
                            m2 = ft[index_t,:].shape[0]
                            X = torch.cat((fs[index_s,:],ft[index_t,:]),dim=0)
                            D = torch.zeros(X.shape[0],2)
                            D[0:m1,0] = 1
                            D[m1:X.shape[0],1] = 1
                            I_c = I_c+hsic_regular(X,D)
                        
                        
                       
                        
                        
                        x_u = ft[index_u,:]
                        u = x_u.shape[0]
                        y_u = torch.zeros(u,n_classes).cuda()
                        y_u[:,25] = 1
                        l_u = 25*torch.ones(u).long().cuda()
                        
                        if u!=0:
                            X0 = torch.cat((fs,ft[h_k1,:],x_u),dim=0)
                            Y0 = torch.cat((ys_one_hot,yt_one_hot[h_k1,:],y_u),dim=0)
                            n = Y0.shape[0]
                            por = (1/torch.sum(Y0,dim=0)/n_classes).reshape(-1,1)
                            a = torch.mm(Y0, por)
                            a = a.cuda()
                            fx = torch.cat((X_s,X_t[h_k1,:],X_t[index_u,:]),dim=0)
                            I_xy = n*hsic_regular0(X0,Y0,a)
                            I_xz = n*hsic_regular0(X0,fx,a)
                        else:
                            X0 = torch.cat((fs,ft[h_k1,:]),dim=0)
                            Y0 = torch.cat((ys_one_hot,yt_one_hot[h_k1,:]),dim=0)
                            n = Y0.shape[0]
                            por = (1/torch.sum(Y0,dim=0)/(n_classes-1)).reshape(-1,1)
                            por[25] = 0
                            a = torch.mm(Y0, por)
                            a = a.cuda()
                            fx = torch.cat((X_s,X_t[h_k1,:],X_t[index_u,:]),dim=0)
                            I_xy = n*hsic_regular0(X0,Y0,a)
                            I_xz = n*hsic_regular0(X0,fx,a)
                        
                        #I_xz = hsic_regular(fs, X_s)
                      
                        
                        #==========================================================
                        #                     Loss Part
                        #==========================================================
                        CE_loss = criterionQ_dis(torch.log(pred_s+1e-4), lab_s)
                      
                        if step <= (TarEnt_step - 1):
                                Tar_Ent_loss = torch.zeros(1).squeeze(0).cuda()
                            
                        else:
                            if u==0:
                                Tar_Ent_loss = 0.8e-2*I_xz-0.8e-3*I_xy+lambda00*I_c
                            else:
                                Tar_Ent_loss = 0.8e-2*I_xz-0.8e-3*I_xy+lambda00*I_c+lambda0*criterionQ_dis(torch.log(pred_t[index_u,:]+1e-4),l_u )
                 
                                    
                        # print(por)
                        #================ Taret Entropy Loss ==============
            
                        #================ Final Objective ==============
                        c_loss = CE_loss + Tar_Ent_loss
                        #================ Final Objective  ==============
            
                        c_loss.backward()
                        c_solver.step()
                       
                        c.eval()
            
                        Test_start_time = datetime.now()
                        print('========================== %1s | Testing start! ==========================='%(Test_start_time))
                        [val_acc,OS,Unk] = classification_accuracy()
            #            train_acc = classification_accuracy_train()
                        train_acc = 0
                        
                        Current_TarEnt_loss = Tar_Ent_loss.cpu().detach().numpy()
                        Current_loss = c_loss.cpu().detach().numpy()    
                        result[:,step] = [1,1,1,Current_loss,train_acc,val_acc,OS,Unk]
            
                        #====================== Time =====================
                        epoch_time_end = datetime.now()
                        seconds = (epoch_time_end - epoch_time_start).seconds
                        minutes = seconds//60
                        second = seconds%60
                        hours = minutes//60
                        minute = minutes%60
                        
                        print('====================== %1s→%1s: Experiment %1s Epoch %1s ================='%(source_domain,target_domain,Exp_iter+1,step+1))

                        print('Train accuracy: {}'.format(train_acc))
                        print('TarEnt_Loss: {}'.format(Current_TarEnt_loss))
                        print('Total_Loss: {}'.format(Current_loss))
                        print('Current epoch time cost: %1s Hour %1s Minutes %1s Seconds'%(hours,minute,second))
                        if max(result[5,:]) == 1:
                            print('Reach accuracy {1} at Epoch %1s !'%(step+1))
                            break
                        
                    #print('==============================Experiment Finish==================')
                    seconds = (epoch_time_end - Exp_start_time).seconds
                    minutes = seconds//60
                    second = seconds%60
                    hours = minutes//60
                    minute = minutes%60
                    Total_Result[:,:,Exp_iter] = result
		    print('Current Max Val_Accuracy: {}'.format(max(result[5,:])))
                    index = np.argmax(result[5,:])
                    Max_ACC[Exp_iter,Domain_iter] = result[5,index]
                    Max_ACC[Exp_iter,Domain_iter+1] = result[6,index]
                    Max_ACC[Exp_iter,Domain_iter+2] = result[7,index]
                    print('Starting TIme: {}'.format(Exp_start_time))
                    print("Finishing TIme: {}".format(epoch_time_end))
                    print('Total TIme Cost: %1s Hour %1s Minutes %1s Seconds'%(hours,minute,second))
                    print("************ %1s→%1s: %1s End Experiment %1s training ************"%(source_domain,target_domain,epoch_time_end,Exp_iter+1))
                    Exp_iter += 1
                Max_ACC[Exptimes,:] = np.mean(Max_ACC[0:Exptimes,:],0)
                print(Max_ACC)
                # result_save_path = base_path+'/lambda0{}_lambda00{}julei_Ixz'.format(lambda0,lambda00)
                Detail_result = Detail_result + (Total_Result,)
                # np.savetxt(result_save_path + f'ACC_list.txt',Max_ACC,delimiter='\t',fmt='%.5f')



