# -*- coding: utf-8 -*-

import argparse
from openpyxl.workbook import Workbook
from lifelines.utils import concordance_index as ci
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import KFold
from lifelines.utils import concordance_index
import os
# %matplotlib inline
import matplotlib.pyplot as plt
import random
from torch.autograd import Variable
from operator import add
import time
import json
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

t1 = time.time()
have_save = True
wd_list = [0.01, 0.001, 0.0001]
wd_index = 2  # 0,1,2


wd = wd_list[wd_index]  # 权重衰减（L2正则化）系数，用于防止过拟合

momentum = 0.7  # 动量参数，可以提高模型在优化过程中的稳定性和收敛速度


lr_dict = dict()
num_dict = dict()
pre_num_list =dict()

lr_dict["brca"] = 0.001
lr_dict["cesc"] = 0.0001
lr_dict["ov"]   = 0.001
lr_dict["ucec"] = 0.001

pre_num_list["brca"] = 200
pre_num_list["cesc"] = 500
pre_num_list["ov"]   = 200
pre_num_list["ucec"] = 100

num_dict["brca"] = 1000
num_dict["cesc"] = 1000
num_dict["ov"]   = 800
num_dict["ucec"] = 1000

folder_name = 'output'  
# 获取当前工作目录  
current_directory = os.getcwd()  

# 构建完整的文件夹路径  
folder_path = os.path.join(current_directory, folder_name)  

# 检查文件夹是否存在  
if not os.path.exists(folder_path):  
    # 如果不存在，则创建文件夹  
    os.makedirs(folder_path)


    
    
def main(cancer_type):  
    if cancer_type in ["brca", "cesc", "ucec", "ov"]:  
        print(f"Processing cancer type: {cancer_type}")  
    else:  
        print("Invalid cancer type. Please use one of 'brca', 'cesc', 'ucec', 'ov'.")  
        return
    
    lr = lr_dict[cancer_type]
    transfer_num=pre_num_list[cancer_type]
    num = num_dict[cancer_type]
    
    samples =["brca_2","cesc_2","ov_2","ucec_2"]
    target = cancer_type+"_2"
   
    filename = [ i for i in samples if i != target]
    save_filename = "+".join(filename)+f"+{transfer_num}"
    print(save_filename)
    
    df = pd.read_csv(f"data//{target}.csv")
    print(list(df.loc[1,:]).count(1))
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
    for train_index, test_index in kf.split(x1):
        yy += 1
        # 划分训练集和测试集
        x1_train, x2_train, y_train = x1_[train_index], x2_[train_index], y_true[train_index]
    
        x1_test, x2_test, y_test, = x1_[test_index], x2_[test_index], y_true[test_index]
        # 定义模型和优化器

        model_ = MMTM_mitigate(nm1, nm2, 4)
        model_.load_state_dict(torch.load(f"model//{save_filename}.pth"))
        optimizer = torch.optim.SGD(model_.parameters(),
                                        lr=lr,
                                        weight_decay=wd,
                                        momentum=momentum)
    
        y_train_ = torch.from_numpy(y_train.copy())
        y_test_ = torch.from_numpy(y_test.copy())
        print(y_train_.shape)
        # 训练模型
        print(y_train_)
        for epoch in range(num):
            model_.train()
            x1_change,x2_change,hazards = model_(x1_train,x2_train)
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
            model_.eval()
            with torch.no_grad():
                _,_,hazards = model_(x1_test,x2_test)
                try :
                    cindex = concordance_index(y_test, -hazards.cpu().detach().numpy(),
                                               )
                    cindex_list2[yy].append(cindex)
                    print(epoch, loss.item(), time.time() - t1, cindex)
                    _,_,hazards = model_(x1_train,x2_train)
        
                    cindex = concordance_index(y_train, -hazards.cpu().detach().numpy())
                    print(epoch, cindex)
                except:
                    num=epoch
                    break
                
            #cindex_list2[yy].append(cindex)
        

        columns = ["epoch", "C-Index"]
        xy = [[i + 1, cindex_list2[yy][i]] for i in range(len(cindex_list2[yy]))]
        df = pd.DataFrame(xy, columns=columns)
        df.to_excel(f"output//{cancer_type}_model{yy}.xlsx", index=False)
    
    # plt.plot([i for i in range(len(loss2)) if i%50 ==0],[loss2[i] for i in range(len(loss2)) if i%50 ==0])
    
    cindex_list2 = np.array(cindex_list2)
    cindex_list2 = cindex_list2.T
    
    
    columns = ["epoch", "C-Index"]
    xy = [[i + 1, cindex_list2[i].mean()] for i in range(len(cindex_list2))]
    df = pd.DataFrame(xy, columns=columns)
    df.to_excel(f"output//{cancer_type}_avg.xlsx", index=False)


if __name__ == "__main__":  
    parser = argparse.ArgumentParser(description='Process cancer type.')  
    parser.add_argument('cancer_type', type=str, default='brca', nargs='?', help='The type of cancer to process. Default is "brca".')  
    args = parser.parse_args()
    main(args.cancer_type)