import  pandas as pd

from lifelines.utils import concordance_index as ci
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import KFold
from lifelines.utils import concordance_index
import sys
import os
import numpy as np
# %matplotlib inline
import matplotlib.pyplot as plt
import torch
import random
from torch import nn
from torch.autograd import Variable
import pandas as pd
from operator import add
import time
import argparse
import json
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import KFold
from lifelines.utils import concordance_index
import pandas as pd
from lifelines import CoxPHFitter
from lifelines import KaplanMeierFitter

from lifelines.utils import concordance_index
import pandas as pd
from lifelines import CoxPHFitter
import torch
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
import torchvision.models as models
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from itertools import combinations
import pandas as pd

import time

from MMTM import MMTM_mitigate
from utils import CoxRegression,concordance_index,DeepCox_Loss





# 设置随机种子
seed = 43
torch.manual_seed(seed)


folder_name = 'model'  

# 获取当前工作目录  
current_directory = os.getcwd()  

# 构建完整的文件夹路径  
folder_path = os.path.join(current_directory, folder_name)  

# 检查文件夹是否存在  
if not os.path.exists(folder_path):  
    # 如果不存在，则创建文件夹  
    os.makedirs(folder_path)




t1 = time.time()

device = torch.device("cpu")

lr_dict = dict()
num_dict = dict()

lr_dict["brca"] = 0.001
lr_dict["cesc"] = 0.001
lr_dict["ov"]   = 0.001
lr_dict["ucec"] = 0.001

num_dict["brca"] = 200
num_dict["cesc"] = 500
num_dict["ov"]   = 200
num_dict["ucec"] = 100

wd_list = [0.01, 0.001, 0.0001]
wd_index = 2  # 0,1,2
wd = wd_list[wd_index]  # 权重衰减（L2正则化）系数，用于防止过拟合

momentum = 0.7  # 动量参数，可以提高模型在优化过程中的稳定性和收敛速度

def main(cancer_type):  
    if cancer_type in ["brca", "cesc", "ucec", "ov"]:  
        print(f"Processing cancer type: {cancer_type}")  
    else:  
        print("Invalid cancer type. Please use one of 'brca', 'cesc', 'ucec', 'ov'.")  
        return
    lr = lr_dict[cancer_type]
    num = num_dict[cancer_type]
    samples =["brca_2","cesc_2","ov_2","ucec_2"]
    target = cancer_type+"_2"
    filename = [ i for i in samples if i != target]
    
    save_filename = "+".join(filename)+f"+{num}"
    
    print(save_filename)
    merged_df = df = pd.read_csv(f"data\\{filename[0]}.csv")
    
    print(merged_df.shape)
    for i in filename[1:]:
        df = pd.read_csv(f"data\\{i}.csv")
        df = df.iloc[:,2:]
        print(df.shape)
        merged_df =pd.concat([merged_df,df], axis=1)
    
    print(list(merged_df.columns).count("GeneSymbol"),list(merged_df.columns).count("Platform"))
    print(merged_df.shape)
    
    df =  merged_df

    nm1 = df.value_counts(["Platform"]).loc["geneExp"].values[0]
    nm2 = df.value_counts(["Platform"]).loc["copyNumber"].values[0]
    
    x1 = df[(df["Platform"] == "geneExp") | (df["Platform"].isna())][:nm1 + 2].T
    x2 = df[(df["Platform"] == "copyNumber") | (df["Platform"].isna())][:nm2 + 2].T
    
    x1.columns = x1.iloc[0]
    x2.columns = x2.iloc[0]
    
    x2.drop("GeneSymbol", axis=0, inplace=True)
    x2.drop("Platform", axis=0, inplace=True)
    x2.drop(["time", "status"], axis=1, inplace=True)
    x1.drop("GeneSymbol", axis=0, inplace=True)
    x1.drop("Platform", axis=0, inplace=True)
    
    # 此时x1和x2的基因名字相同，要分开
    s = {}
    f = {}
    f2 = {}
    for i in x2.columns:
        s[i] = i + "_2"
        f[i + "_2"] = "float"
    x2.rename(columns=s, inplace=True)
    x2 = x2.astype(f)
    
    for i in x1.columns:
        f2[i] = "float"
    x1 = x1.astype(f2)
    
    x = pd.concat([x1, x2], axis=1)
    
    x = x.fillna(x.median())
    x = x[x["time"] >= 0]
    
    # 加载数据
    x1 = np.array(x.iloc[:, 2:nm1 + 2])
    x2 = np.array(x.iloc[:, nm1 + 2:])
    x1 = torch.from_numpy(x1).float()
    x2 = torch.from_numpy(x2).float()
    
    y = np.array(x.iloc[:, 0])
    y = torch.from_numpy(y).float().reshape(-1, 1)

    loss2 = []
    scaler1 = MinMaxScaler()
    scaler2 = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    x1_ = scaler1.fit_transform(x1)
    x2_ = scaler2.fit_transform(x2)
    
    x1_ = torch.from_numpy(x1_).float()
    x2_ = torch.from_numpy(x2_).float()
    
    y_ = scaler_y.fit_transform(y)
    y_ = torch.from_numpy(y_).float()
    
    y_true = x.iloc[:, 0].to_list()
    for i in range(len(y_true)):
        if x.iloc[:, 1][i] == 0:
            y_true[i] = -y_true[i]
    y_true = np.array(y_true)
    
    cindex = []
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)

    cox_loss = DeepCox_Loss()
    # 开始交叉验证
    batch_size = 32
    cindex_list = []
    
    yy = -1
    cindex_list2 = [[] for i in range(5)]
    
    
    
    x1_train, x2_train, y_train = x1_, x2_, y_true
    
    transfer_model = MMTM_mitigate(nm1, nm2, 4)
    
    optimizer = torch.optim.SGD(transfer_model.parameters(),
                                        lr=lr,
                                        weight_decay=wd,
                                        momentum=momentum)
    
    y_train_ = torch.from_numpy(y_train.copy())
    
    print(y_train_.shape)
    # 训练模型
    print(y_train_)
    for epoch in range(num):
            transfer_model.train()
            x1_change,x2_change,hazards = transfer_model(x1_train,x2_train)
            # 转换为DataFrame
            loss_y = cox_loss(hazards, y_train_)
            reconstruction_loss1 = nn.functional.mse_loss(x1_change,x1_train,reduction="sum")
            reconstruction_loss2= nn.functional.mse_loss(x2_change, x2_train, reduction="sum")
            optimizer.zero_grad()
            loss = loss_y+reconstruction_loss1+reconstruction_loss2
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model2.parameters(), 2)
            # torch.nn.utils.clip_grad_norm_(model2.cox.parameters(), 1)
            optimizer.step()
            # 在每个epoch结束后计算C-index
            transfer_model.eval()
            with torch.no_grad():
                _,_,hazards = transfer_model(x1_train,x2_train)
                cindex = concordance_index(y_train, -hazards.cpu().detach().numpy())
                print(epoch, loss.item(), time.time() - t1,cindex)
            cindex_list.append(cindex)
    
    # plt.plot([i for i in range(len(loss2)) if i%50 ==0],[loss2[i] for i in range(len(loss2)) if i%50 ==0])
    
    #fig = plt.figure()
    xmm = [i for i in range(num) if i % 5 == 0]
    #fig.gca().set_title("C-INDEX")
    #plt.plot(xmm, [cindex_list[i] for i in xmm])
    #plt.savefig(f"{save_filename}.png", dpi=400)
    
    columns = ["epoch", "C-Index"]
    xy = [[i + 1, cindex_list[i]] for i in range(len(cindex_list))]
    #df = pd.DataFrame(xy, columns=columns)
    #df.to_excel(f"{save_filename}.xlsx", index=False)

    torch.save(transfer_model.state_dict(), f"model//{save_filename}.pth")
    
    plt.show()

    
if __name__ == "__main__":  
    parser = argparse.ArgumentParser(description='Process cancer type.')  
    parser.add_argument('cancer_type', type=str, default='brca', nargs='?', help='The type of cancer to process. Default is "brca".')  
    args = parser.parse_args()  
  
    main(args.cancer_type)





