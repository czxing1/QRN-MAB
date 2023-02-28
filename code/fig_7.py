##头文件

from cmath import log, sqrt
import random
import matplotlib.pyplot as plt
import numpy as np 
import math 
from scipy.stats import bernoulli
import itertools as it
import csv 


# 类混沌生成
chaos_num = 0.7
iters = 10000
chaos_list = []
for i in range(1,iters):
    chaos_num = math.cos(i*math.cos(chaos_num)**(-1))/2
    chaos_list.append(chaos_num)




# 信道检测函数
def channel_test():
    R1 = random.random()
    if R1 < 0.3:
        result = 0
    else:
        result = 1
    return result

##初始化参数
M = 1
#信道数量
N = 10
restart = True
##定位M值
while(restart):
    if(2**M < N < 2**(M+1)):
        restart = False
    M += 1
loss_num = 2**M - N

T = 5000
alpha = 0.99
Z = 128
delta = 0.1
Bit = 100
#CFL算法中的迭代次数
iteration = 3
#每个time循环次数
epochs = 5
##DSSL算法参数
epsilon = 1
L = 2
NI = 1
I = 7*epsilon**(2)/(48*L*(1+2)**(2))
t = 0



##初始化用户
users = ['a','b','c','d','e','f','g','h']
##初始化阈值
Threshold_1 = {
    'a':{'TH1':0, 'TH2_0':0, 'TH2_1':0, 'TH3_00':0, 'TH3_01':0, 'TH3_10':0, 'TH3_11':0, 'TH4_111':0, 'TH4_110':0, 'TH4_101':0, 'TH4_100':0, 'TH4_011':0, 'TH4_010':0, 'TH4_001':0, 'TH4_000':0, 'ome':1, 'lam':1},
    'b':{'TH1':0, 'TH2_0':0, 'TH2_1':0, 'TH3_00':0, 'TH3_01':0, 'TH3_10':0, 'TH3_11':0, 'TH4_111':0, 'TH4_110':0, 'TH4_101':0, 'TH4_100':0, 'TH4_011':0, 'TH4_010':0, 'TH4_001':0, 'TH4_000':0, 'ome':1, 'lam':1},
    'c':{'TH1':0, 'TH2_0':0, 'TH2_1':0, 'TH3_00':0, 'TH3_01':0, 'TH3_10':0, 'TH3_11':0, 'TH4_111':0, 'TH4_110':0, 'TH4_101':0, 'TH4_100':0, 'TH4_011':0, 'TH4_010':0, 'TH4_001':0, 'TH4_000':0, 'ome':1, 'lam':1},
    'd':{'TH1':0, 'TH2_0':0, 'TH2_1':0, 'TH3_00':0, 'TH3_01':0, 'TH3_10':0, 'TH3_11':0, 'TH4_111':0, 'TH4_110':0, 'TH4_101':0, 'TH4_100':0, 'TH4_011':0, 'TH4_010':0, 'TH4_001':0, 'TH4_000':0, 'ome':1, 'lam':1},
    'e':{'TH1':0, 'TH2_0':0, 'TH2_1':0, 'TH3_00':0, 'TH3_01':0, 'TH3_10':0, 'TH3_11':0, 'TH4_111':0, 'TH4_110':0, 'TH4_101':0, 'TH4_100':0, 'TH4_011':0, 'TH4_010':0, 'TH4_001':0, 'TH4_000':0, 'ome':1, 'lam':1},
    'f':{'TH1':0, 'TH2_0':0, 'TH2_1':0, 'TH3_00':0, 'TH3_01':0, 'TH3_10':0, 'TH3_11':0, 'TH4_111':0, 'TH4_110':0, 'TH4_101':0, 'TH4_100':0, 'TH4_011':0, 'TH4_010':0, 'TH4_001':0, 'TH4_000':0, 'ome':1, 'lam':1},
    'g':{'TH1':0, 'TH2_0':0, 'TH2_1':0, 'TH3_00':0, 'TH3_01':0, 'TH3_10':0, 'TH3_11':0, 'TH4_111':0, 'TH4_110':0, 'TH4_101':0, 'TH4_100':0, 'TH4_011':0, 'TH4_010':0, 'TH4_001':0, 'TH4_000':0, 'ome':1, 'lam':1},
    'h':{'TH1':0, 'TH2_0':0, 'TH2_1':0, 'TH3_00':0, 'TH3_01':0, 'TH3_10':0, 'TH3_11':0, 'TH4_111':0, 'TH4_110':0, 'TH4_101':0, 'TH4_100':0, 'TH4_011':0, 'TH4_010':0, 'TH4_001':0, 'TH4_000':0, 'ome':1, 'lam':1},
}
Threshold_2 = {
    'a':{'TH1':0, 'TH2_0':0, 'TH2_1':0, 'TH3_00':0, 'TH3_01':0, 'TH3_10':0, 'TH3_11':0, 'TH4_111':0, 'TH4_110':0, 'TH4_101':0, 'TH4_100':0, 'TH4_011':0, 'TH4_010':0, 'TH4_001':0, 'TH4_000':0, 'ome':1, 'lam':1},
    'b':{'TH1':0, 'TH2_0':0, 'TH2_1':0, 'TH3_00':0, 'TH3_01':0, 'TH3_10':0, 'TH3_11':0, 'TH4_111':0, 'TH4_110':0, 'TH4_101':0, 'TH4_100':0, 'TH4_011':0, 'TH4_010':0, 'TH4_001':0, 'TH4_000':0, 'ome':1, 'lam':1},
    'c':{'TH1':0, 'TH2_0':0, 'TH2_1':0, 'TH3_00':0, 'TH3_01':0, 'TH3_10':0, 'TH3_11':0, 'TH4_111':0, 'TH4_110':0, 'TH4_101':0, 'TH4_100':0, 'TH4_011':0, 'TH4_010':0, 'TH4_001':0, 'TH4_000':0, 'ome':1, 'lam':1},
    'd':{'TH1':0, 'TH2_0':0, 'TH2_1':0, 'TH3_00':0, 'TH3_01':0, 'TH3_10':0, 'TH3_11':0, 'TH4_111':0, 'TH4_110':0, 'TH4_101':0, 'TH4_100':0, 'TH4_011':0, 'TH4_010':0, 'TH4_001':0, 'TH4_000':0, 'ome':1, 'lam':1},
    'e':{'TH1':0, 'TH2_0':0, 'TH2_1':0, 'TH3_00':0, 'TH3_01':0, 'TH3_10':0, 'TH3_11':0, 'TH4_111':0, 'TH4_110':0, 'TH4_101':0, 'TH4_100':0, 'TH4_011':0, 'TH4_010':0, 'TH4_001':0, 'TH4_000':0, 'ome':1, 'lam':1},
    'f':{'TH1':0, 'TH2_0':0, 'TH2_1':0, 'TH3_00':0, 'TH3_01':0, 'TH3_10':0, 'TH3_11':0, 'TH4_111':0, 'TH4_110':0, 'TH4_101':0, 'TH4_100':0, 'TH4_011':0, 'TH4_010':0, 'TH4_001':0, 'TH4_000':0, 'ome':1, 'lam':1},
    'g':{'TH1':0, 'TH2_0':0, 'TH2_1':0, 'TH3_00':0, 'TH3_01':0, 'TH3_10':0, 'TH3_11':0, 'TH4_111':0, 'TH4_110':0, 'TH4_101':0, 'TH4_100':0, 'TH4_011':0, 'TH4_010':0, 'TH4_001':0, 'TH4_000':0, 'ome':1, 'lam':1},
    'h':{'TH1':0, 'TH2_0':0, 'TH2_1':0, 'TH3_00':0, 'TH3_01':0, 'TH3_10':0, 'TH3_11':0, 'TH4_111':0, 'TH4_110':0, 'TH4_101':0, 'TH4_100':0, 'TH4_011':0, 'TH4_010':0, 'TH4_001':0, 'TH4_000':0, 'ome':1, 'lam':1},
}
Threshold_3 = {
    'a':{'TH1':0, 'TH2_0':0, 'TH2_1':0, 'TH3_00':0, 'TH3_01':0, 'TH3_10':0, 'TH3_11':0, 'TH4_111':0, 'TH4_110':0, 'TH4_101':0, 'TH4_100':0, 'TH4_011':0, 'TH4_010':0, 'TH4_001':0, 'TH4_000':0, 'ome':1, 'lam':1},
    'b':{'TH1':0, 'TH2_0':0, 'TH2_1':0, 'TH3_00':0, 'TH3_01':0, 'TH3_10':0, 'TH3_11':0, 'TH4_111':0, 'TH4_110':0, 'TH4_101':0, 'TH4_100':0, 'TH4_011':0, 'TH4_010':0, 'TH4_001':0, 'TH4_000':0, 'ome':1, 'lam':1},
    'c':{'TH1':0, 'TH2_0':0, 'TH2_1':0, 'TH3_00':0, 'TH3_01':0, 'TH3_10':0, 'TH3_11':0, 'TH4_111':0, 'TH4_110':0, 'TH4_101':0, 'TH4_100':0, 'TH4_011':0, 'TH4_010':0, 'TH4_001':0, 'TH4_000':0, 'ome':1, 'lam':1},
    'd':{'TH1':0, 'TH2_0':0, 'TH2_1':0, 'TH3_00':0, 'TH3_01':0, 'TH3_10':0, 'TH3_11':0, 'TH4_111':0, 'TH4_110':0, 'TH4_101':0, 'TH4_100':0, 'TH4_011':0, 'TH4_010':0, 'TH4_001':0, 'TH4_000':0, 'ome':1, 'lam':1},
    'e':{'TH1':0, 'TH2_0':0, 'TH2_1':0, 'TH3_00':0, 'TH3_01':0, 'TH3_10':0, 'TH3_11':0, 'TH4_111':0, 'TH4_110':0, 'TH4_101':0, 'TH4_100':0, 'TH4_011':0, 'TH4_010':0, 'TH4_001':0, 'TH4_000':0, 'ome':1, 'lam':1},
    'f':{'TH1':0, 'TH2_0':0, 'TH2_1':0, 'TH3_00':0, 'TH3_01':0, 'TH3_10':0, 'TH3_11':0, 'TH4_111':0, 'TH4_110':0, 'TH4_101':0, 'TH4_100':0, 'TH4_011':0, 'TH4_010':0, 'TH4_001':0, 'TH4_000':0, 'ome':1, 'lam':1},
    'g':{'TH1':0, 'TH2_0':0, 'TH2_1':0, 'TH3_00':0, 'TH3_01':0, 'TH3_10':0, 'TH3_11':0, 'TH4_111':0, 'TH4_110':0, 'TH4_101':0, 'TH4_100':0, 'TH4_011':0, 'TH4_010':0, 'TH4_001':0, 'TH4_000':0, 'ome':1, 'lam':1},
    'h':{'TH1':0, 'TH2_0':0, 'TH2_1':0, 'TH3_00':0, 'TH3_01':0, 'TH3_10':0, 'TH3_11':0, 'TH4_111':0, 'TH4_110':0, 'TH4_101':0, 'TH4_100':0, 'TH4_011':0, 'TH4_010':0, 'TH4_001':0, 'TH4_000':0, 'ome':1, 'lam':1},
}
Threshold_4= {
    'a':{'TH1':0, 'TH2_0':0, 'TH2_1':0, 'TH3_00':0, 'TH3_01':0, 'TH3_10':0, 'TH3_11':0, 'TH4_111':0, 'TH4_110':0, 'TH4_101':0, 'TH4_100':0, 'TH4_011':0, 'TH4_010':0, 'TH4_001':0, 'TH4_000':0, 'ome':1, 'lam':1},
    'b':{'TH1':0, 'TH2_0':0, 'TH2_1':0, 'TH3_00':0, 'TH3_01':0, 'TH3_10':0, 'TH3_11':0, 'TH4_111':0, 'TH4_110':0, 'TH4_101':0, 'TH4_100':0, 'TH4_011':0, 'TH4_010':0, 'TH4_001':0, 'TH4_000':0, 'ome':1, 'lam':1},
    'c':{'TH1':0, 'TH2_0':0, 'TH2_1':0, 'TH3_00':0, 'TH3_01':0, 'TH3_10':0, 'TH3_11':0, 'TH4_111':0, 'TH4_110':0, 'TH4_101':0, 'TH4_100':0, 'TH4_011':0, 'TH4_010':0, 'TH4_001':0, 'TH4_000':0, 'ome':1, 'lam':1},
    'd':{'TH1':0, 'TH2_0':0, 'TH2_1':0, 'TH3_00':0, 'TH3_01':0, 'TH3_10':0, 'TH3_11':0, 'TH4_111':0, 'TH4_110':0, 'TH4_101':0, 'TH4_100':0, 'TH4_011':0, 'TH4_010':0, 'TH4_001':0, 'TH4_000':0, 'ome':1, 'lam':1},
    'e':{'TH1':0, 'TH2_0':0, 'TH2_1':0, 'TH3_00':0, 'TH3_01':0, 'TH3_10':0, 'TH3_11':0, 'TH4_111':0, 'TH4_110':0, 'TH4_101':0, 'TH4_100':0, 'TH4_011':0, 'TH4_010':0, 'TH4_001':0, 'TH4_000':0, 'ome':1, 'lam':1},
    'f':{'TH1':0, 'TH2_0':0, 'TH2_1':0, 'TH3_00':0, 'TH3_01':0, 'TH3_10':0, 'TH3_11':0, 'TH4_111':0, 'TH4_110':0, 'TH4_101':0, 'TH4_100':0, 'TH4_011':0, 'TH4_010':0, 'TH4_001':0, 'TH4_000':0, 'ome':1, 'lam':1},
    'g':{'TH1':0, 'TH2_0':0, 'TH2_1':0, 'TH3_00':0, 'TH3_01':0, 'TH3_10':0, 'TH3_11':0, 'TH4_111':0, 'TH4_110':0, 'TH4_101':0, 'TH4_100':0, 'TH4_011':0, 'TH4_010':0, 'TH4_001':0, 'TH4_000':0, 'ome':1, 'lam':1},
    'h':{'TH1':0, 'TH2_0':0, 'TH2_1':0, 'TH3_00':0, 'TH3_01':0, 'TH3_10':0, 'TH3_11':0, 'TH4_111':0, 'TH4_110':0, 'TH4_101':0, 'TH4_100':0, 'TH4_011':0, 'TH4_010':0, 'TH4_001':0, 'TH4_000':0, 'ome':1, 'lam':1},
}
##初始化(激光混沌)发起者数目
init_num = 1
init_num = np.clip(init_num,1,len(users))

##初始化用户对每个信道的概率
priv_prob_1 = [
    [[0.9,0,0],[0.8,0,0],[0.7,0,0],[0.6,0,0],[0.4,0,0],[0.4,0,0],[0.3,0,0],[0.7,0,0],[0.9,0,0],[0.8,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]],
    [[0.7,0,0],[0.6,0,0],[0.5,0,0],[0.4,0,0],[0.2,0,0],[0.2,0,0],[0.9,0,0],[0.8,0,0],[0.9,0,0],[0.8,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]],
    [[0.2,0,0],[0.5,0,0],[0.2,0,0],[0.9,0,0],[0.8,0,0],[0.8,0,0],[0.6,0,0],[0.5,0,0],[0.3,0,0],[0.1,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]],
    [[0.8,0,0],[0.9,0,0],[0.6,0,0],[0.5,0,0],[0.1,0,0],[0.3,0,0],[0.4,0,0],[0.7,0,0],[0.1,0,0],[0.9,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]],
    [[0.4,0,0],[0.3,0,0],[0.8,0,0],[0.1,0,0],[0.3,0,0],[0.8,0,0],[0.9,0,0],[0.1,0,0],[0.5,0,0],[0.2,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]],
    [[0.2,0,0],[0.9,0,0],[0.4,0,0],[0.8,0,0],[0.5,0,0],[0.5,0,0],[0.4,0,0],[0.1,0,0],[0.3,0,0],[0.2,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]],
    [[0.7,0,0],[0.6,0,0],[0.2,0,0],[0.8,0,0],[0.9,0,0],[0.4,0,0],[0.3,0,0],[0.2,0,0],[0.1,0,0],[0.7,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]],
    [[0.6,0,0],[0.5,0,0],[0.4,0,0],[0.3,0,0],[0.2,0,0],[0.1,0,0],[0.8,0,0],[0.9,0,0],[0.7,0,0],[0.6,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]],
]




##初始化

## 初始化信道
trans_rvs = list(it.product(range(2),repeat = M)) 
signalpaths = []
for trans_rv in trans_rvs:
    res = []
    for rv in trans_rv:
        res.append('%d'%rv)
    trans = ''.join(res)
    signalpaths.append(trans)
init_path = signalpaths[0:N]
signalpaths_DSSL = signalpaths[0:N]

# loss_paths = signalpaths[-loss_num:]

##初始化激光混沌用户平均收益和偏好
priv_1 = {};priv_2 = {};priv_3 = {};priv_4 = {}

ave_earn = [0]*len(signalpaths)

for user in users:
    priv_1[user] = {'path':'','pref':[]}
    priv_1[user]['ave_earn'] = dict(zip(signalpaths,ave_earn))

    priv_2[user] = {'path':'','pref':[]}
    priv_2[user]['ave_earn'] = dict(zip(signalpaths,ave_earn))

    priv_3[user] = {'path':'','pref':[]}
    priv_3[user]['ave_earn'] = dict(zip(signalpaths,ave_earn))

    priv_4[user] = {'path':'','pref':[]}
    priv_4[user]['ave_earn'] = dict(zip(signalpaths,ave_earn))
##初始化各个用户通过信道的概率
signal_path_1 = {};signal_path_2 = {};signal_path_3 = {};signal_path_4 = {}
index = 0
for user in users:
    signal_path_1[user] = dict(zip(signalpaths,priv_prob_1[index]))
    signal_path_2[user] = dict(zip(signalpaths,priv_prob_1[index]))
    signal_path_3[user] = dict(zip(signalpaths,priv_prob_1[index]))
    signal_path_4[user] = dict(zip(signalpaths,priv_prob_1[index]))
    
    index = index + 1



##初始化信道选择以及偏好


for user in users:
    
    #用户以概率p选择信道
    p = [1/len(init_path)]*len(init_path)    
    # 初始化信道概率
    p_accu = list(np.divide(range(0,len(init_path)+1),len(init_path)))

    # 进行iteration次迭代
    for m in range(0,iteration):
        # 以概率p选择信道，根据信道结果更新概率
        R = random.random()
        for i in range(0,len(init_path)):
            if (p_accu[i] < R < p_accu[i+1]):
                
                channel = init_path[i]
                #更新平均收益

                earn = bernoulli.rvs(signal_path_1[user][channel][0])
                signal_path_1[user][channel][1] = signal_path_1[user][channel][1] + 1 
                signal_path_1[user][channel][2] = signal_path_1[user][channel][2] + earn
                priv_1[user]['ave_earn'][channel] = signal_path_1[user][channel][2]/signal_path_1[user][channel][1]

                earn = bernoulli.rvs(signal_path_2[user][channel][0])
                signal_path_2[user][channel][1] = signal_path_2[user][channel][1] + 1 
                signal_path_2[user][channel][2] = signal_path_2[user][channel][2] + earn
                priv_2[user]['ave_earn'][channel] = signal_path_2[user][channel][2]/signal_path_2[user][channel][1]

                earn = bernoulli.rvs(signal_path_3[user][channel][0])
                signal_path_3[user][channel][1] = signal_path_3[user][channel][1] + 1 
                signal_path_3[user][channel][2] = signal_path_3[user][channel][2] + earn
                priv_3[user]['ave_earn'][channel] = signal_path_3[user][channel][2]/signal_path_3[user][channel][1]

                earn = bernoulli.rvs(signal_path_4[user][channel][0])
                signal_path_4[user][channel][1] = signal_path_4[user][channel][1] + 1 
                signal_path_4[user][channel][2] = signal_path_4[user][channel][2] + earn
                priv_4[user]['ave_earn'][channel] = signal_path_4[user][channel][2]/signal_path_4[user][channel][1]
                result = channel_test()
                if(result == 1):
                    p = [0]*len(init_path)
                    p[i] = 1
                else:
                    p = [(x*0.9 + 0.1/(len(init_path) - 1)) for x in p]
                    p[i] = p[i] - 0.1/(len(init_path) - 1)
                # 更新p_accu, 存储本次选择的信道, 存储每一次的结果
                accu = 0
                for k in range(0,len(init_path)):
                    accu += p[k]
                    p_accu[k+1] = accu
                
                break
            else:
                continue
    seq1 = [];seq2 = [];seq3 = [];seq4 = []
    prefs_1 = sorted(priv_1[user]['ave_earn'].items(),key = lambda x:x[1],reverse= True)
    prefs_2 = sorted(priv_2[user]['ave_earn'].items(),key = lambda x:x[1],reverse= True)
    prefs_3 = sorted(priv_3[user]['ave_earn'].items(),key = lambda x:x[1],reverse= True)
    prefs_4 = sorted(priv_4[user]['ave_earn'].items(),key = lambda x:x[1],reverse= True)

    for pref in prefs_1:
        seq1.append(pref[0])
    for pref in prefs_2:
        seq2.append(pref[0])
    for pref in prefs_3:
        seq3.append(pref[0])
    for pref in prefs_4:
        seq4.append(pref[0])
   
    priv_1[user]['pref'] = seq1
    priv_1[user]['path'] = channel
    
    priv_2[user]['pref'] = seq2
    priv_2[user]['path'] = channel
    
    priv_3[user]['pref'] = seq3
    priv_3[user]['path'] = channel
    
    priv_4[user]['pref'] = seq4
    priv_4[user]['path'] = channel


##初始化总体收益
total_earns_1 = []
total_earns_2 = []
total_earns_3 = []
total_earns_4 = []

##初始化交换次数
change_times_1 = [0]
change_times_2 = [0]
change_times_3 = [0]
change_times_4 = [0]


# QRN-MAB 算法
for time in range(0,T):

    ## 激光混沌算法只用于更新平均收益以及偏好！
    ## 求epochs次的平均值
    total_earn_epoch_1 = 0
    change_time_epoch_1 = 0

    total_earn_epoch_2 = 0
    change_time_epoch_2 = 0

    total_earn_epoch_3 = 0
    change_time_epoch_3 = 0

    total_earn_epoch_4 = 0
    change_time_epoch_4 = 0


    for epoch in range(0,epochs):
        change_time_1 = 0
        change_time_2 = 0
        change_time_3 = 0
        change_time_4 = 0

        for user in users:

            #这里替换成利用激光生成随机数
            s1 = random.uniform(-Z,Z)
            s2 = random.uniform(-Z,Z)
            s3 = random.uniform(-Z,Z)
            s4 = random.uniform(-Z,Z)

            if(s1>= Threshold_1[user]['TH1']):
                j1 = '1'       
                if(s2>= Threshold_1[user]['TH2_1']):
                    j2 = '1'
                    if(s3>= Threshold_1[user]['TH3_11']):
                        j3 = '1'
                        if(s4>= Threshold_1[user]['TH4_111']):
                            j4 = '1'
                            j = j1+j2+j3+j4
                            earn = bernoulli.rvs(signal_path_1[user][j][0])
                            signal_path_1[user][j][1] = signal_path_1[user][j][1] + 1 
                            signal_path_1[user][j][2] = signal_path_1[user][j][2] + earn
                            #priv_1[user]['path'] = j
                            priv_1[user]['ave_earn'][j] = signal_path_1[user][j][2]/signal_path_1[user][j][1]
                            # priv_fuzz[user]['ave_earn'][j] = signal_path_1[user][j][2]/signal_path_1[user][j][1]
                            if(earn > 0):
                                Threshold_1[user]['TH1'] = alpha*Threshold_1[user]['TH1'] - Threshold_1[user]['lam'] 
                                Threshold_1[user]['TH2_1'] = alpha*Threshold_1[user]['TH2_1'] - Threshold_1[user]['lam']
                                Threshold_1[user]['TH3_11'] = alpha*Threshold_1[user]['TH3_11'] - Threshold_1[user]['lam']
                                Threshold_1[user]['TH4_111'] = alpha*Threshold_1[user]['TH4_111'] - Threshold_1[user]['lam']
                            else:
                                Threshold_1[user]['TH1'] = alpha*Threshold_1[user]['TH1'] + Threshold_1[user]['ome']
                                Threshold_1[user]['TH2_1'] = alpha*Threshold_1[user]['TH2_1'] + Threshold_1[user]['ome']
                                Threshold_1[user]['TH3_11'] = alpha*Threshold_1[user]['TH3_11'] + Threshold_1[user]['ome']
                                Threshold_1[user]['TH4_111'] = alpha*Threshold_1[user]['TH4_111'] + Threshold_1[user]['ome'] 
                            Threshold_1[user]['TH1'] = np.clip(Threshold_1[user]['TH1'],-Z,Z)
                            Threshold_1[user]['TH2_1'] = np.clip(Threshold_1[user]['TH2_1'],-Z,Z)
                            Threshold_1[user]['TH3_11'] = np.clip(Threshold_1[user]['TH3_11'],-Z,Z)
                            Threshold_1[user]['TH4_111'] = np.clip(Threshold_1[user]['TH4_111'],-Z,Z)
                        else:
                            j4 = '0'
                            j = j1+j2+j3+j4
                            earn = bernoulli.rvs(signal_path_1[user][j][0])
                            signal_path_1[user][j][1] = signal_path_1[user][j][1] + 1 
                            signal_path_1[user][j][2] = signal_path_1[user][j][2] + earn
                            #priv_1[user]['path'] = j
                            # priv_fuzz[user]['ave_earn'][j] = signal_path_1[user][j][2]/signal_path_1[user][j][1]
                            priv_1[user]['ave_earn'][j] = signal_path_1[user][j][2]/signal_path_1[user][j][1]
                            if(earn > 0):
                                Threshold_1[user]['TH1'] = alpha*Threshold_1[user]['TH1'] - Threshold_1[user]['lam'] 
                                Threshold_1[user]['TH2_1'] = alpha*Threshold_1[user]['TH2_1'] - Threshold_1[user]['lam']
                                Threshold_1[user]['TH3_11'] = alpha*Threshold_1[user]['TH3_11'] - Threshold_1[user]['lam']
                                Threshold_1[user]['TH4_111'] = alpha*Threshold_1[user]['TH4_111'] + Threshold_1[user]['lam']
                            else:
                                Threshold_1[user]['TH1'] = alpha*Threshold_1[user]['TH1'] + Threshold_1[user]['ome']
                                Threshold_1[user]['TH2_1'] = alpha*Threshold_1[user]['TH2_1'] + Threshold_1[user]['ome']
                                Threshold_1[user]['TH3_11'] = alpha*Threshold_1[user]['TH3_11'] + Threshold_1[user]['ome']
                                Threshold_1[user]['TH4_111'] = alpha*Threshold_1[user]['TH4_111'] - Threshold_1[user]['ome'] 
                            Threshold_1[user]['TH1'] = np.clip(Threshold_1[user]['TH1'],-Z,Z)
                            Threshold_1[user]['TH2_1'] = np.clip(Threshold_1[user]['TH2_1'],-Z,Z)
                            Threshold_1[user]['TH3_11'] = np.clip(Threshold_1[user]['TH3_11'],-Z,Z)
                            Threshold_1[user]['TH4_111'] = np.clip(Threshold_1[user]['TH4_111'],-Z,Z)
                    else:
                        j3 = '0'
                        if(s4>= Threshold_1[user]['TH4_110']):
                            j4 = '1'
                            j = j1+j2+j3+j4
                            earn = bernoulli.rvs(signal_path_1[user][j][0])
                            signal_path_1[user][j][1] = signal_path_1[user][j][1] + 1 
                            signal_path_1[user][j][2] = signal_path_1[user][j][2] + earn
                            #priv_1[user]['path'] = j
                            # priv_fuzz[user]['ave_earn'][j] = signal_path_1[user][j][2]/signal_path_1[user][j][1]
                            priv_1[user]['ave_earn'][j] = signal_path_1[user][j][2]/signal_path_1[user][j][1]
                            if(earn > 0):
                                Threshold_1[user]['TH1'] = alpha*Threshold_1[user]['TH1'] - Threshold_1[user]['lam'] 
                                Threshold_1[user]['TH2_1'] = alpha*Threshold_1[user]['TH2_1'] - Threshold_1[user]['lam']
                                Threshold_1[user]['TH3_11'] = alpha*Threshold_1[user]['TH3_11'] + Threshold_1[user]['lam']
                                Threshold_1[user]['TH4_110'] = alpha*Threshold_1[user]['TH4_110'] - Threshold_1[user]['lam']
                            else:
                                Threshold_1[user]['TH1'] = alpha*Threshold_1[user]['TH1'] + Threshold_1[user]['ome']
                                Threshold_1[user]['TH2_1'] = alpha*Threshold_1[user]['TH2_1'] + Threshold_1[user]['ome']
                                Threshold_1[user]['TH3_11'] = alpha*Threshold_1[user]['TH3_11'] - Threshold_1[user]['ome']
                                Threshold_1[user]['TH4_110'] = alpha*Threshold_1[user]['TH4_110'] + Threshold_1[user]['ome'] 
                            Threshold_1[user]['TH1'] = np.clip(Threshold_1[user]['TH1'],-Z,Z)
                            Threshold_1[user]['TH2_1'] = np.clip(Threshold_1[user]['TH2_1'],-Z,Z)
                            Threshold_1[user]['TH3_11'] = np.clip(Threshold_1[user]['TH3_11'],-Z,Z)
                            Threshold_1[user]['TH4_110'] = np.clip(Threshold_1[user]['TH4_110'],-Z,Z)
                        else:
                            j4 = '0'
                            j = j1+j2+j3+j4
                            earn = bernoulli.rvs(signal_path_1[user][j][0])
                            signal_path_1[user][j][1] = signal_path_1[user][j][1] + 1 
                            signal_path_1[user][j][2] = signal_path_1[user][j][2] + earn
                            #priv_1[user]['path'] = j
                            # priv_fuzz[user]['ave_earn'][j] = signal_path_1[user][j][2]/signal_path_1[user][j][1]
                            priv_1[user]['ave_earn'][j] = signal_path_1[user][j][2]/signal_path_1[user][j][1]
                            if(earn > 0):
                                Threshold_1[user]['TH1'] = alpha*Threshold_1[user]['TH1'] - Threshold_1[user]['lam'] 
                                Threshold_1[user]['TH2_1'] = alpha*Threshold_1[user]['TH2_1'] - Threshold_1[user]['lam']
                                Threshold_1[user]['TH3_11'] = alpha*Threshold_1[user]['TH3_11'] + Threshold_1[user]['lam']
                                Threshold_1[user]['TH4_110'] = alpha*Threshold_1[user]['TH4_110'] + Threshold_1[user]['lam']
                            else:
                                Threshold_1[user]['TH1'] = alpha*Threshold_1[user]['TH1'] + Threshold_1[user]['ome']
                                Threshold_1[user]['TH2_1'] = alpha*Threshold_1[user]['TH2_1'] + Threshold_1[user]['ome']
                                Threshold_1[user]['TH3_11'] = alpha*Threshold_1[user]['TH3_11'] - Threshold_1[user]['ome']
                                Threshold_1[user]['TH4_110'] = alpha*Threshold_1[user]['TH4_110'] - Threshold_1[user]['ome'] 
                            Threshold_1[user]['TH1'] = np.clip(Threshold_1[user]['TH1'],-Z,Z)
                            Threshold_1[user]['TH2_1'] = np.clip(Threshold_1[user]['TH2_1'],-Z,Z)
                            Threshold_1[user]['TH3_11'] = np.clip(Threshold_1[user]['TH3_11'],-Z,Z)
                            Threshold_1[user]['TH4_110'] = np.clip(Threshold_1[user]['TH4_110'],-Z,Z)                     
                else:
                    j2 = '0'
                    if(s3>= Threshold_1[user]['TH3_10']):
                        j3 = '1'
                        if(s4>= Threshold_1[user]['TH4_101']):
                            j4 = '1'
                            j = j1+j2+j3+j4
                            earn = bernoulli.rvs(signal_path_1[user][j][0])
                            signal_path_1[user][j][1] = signal_path_1[user][j][1] + 1 
                            signal_path_1[user][j][2] = signal_path_1[user][j][2] + earn
                            #priv_1[user]['path'] = j
                            # priv_fuzz[user]['ave_earn'][j] = signal_path_1[user][j][2]/signal_path_1[user][j][1]
                            priv_1[user]['ave_earn'][j] = signal_path_1[user][j][2]/signal_path_1[user][j][1]
                            if(earn > 0):
                                Threshold_1[user]['TH1'] = alpha*Threshold_1[user]['TH1'] - Threshold_1[user]['lam'] 
                                Threshold_1[user]['TH2_1'] = alpha*Threshold_1[user]['TH2_1'] + Threshold_1[user]['lam']
                                Threshold_1[user]['TH3_10'] = alpha*Threshold_1[user]['TH3_10'] - Threshold_1[user]['lam']
                                Threshold_1[user]['TH4_101'] = alpha*Threshold_1[user]['TH4_101'] - Threshold_1[user]['lam']
                            else:
                                Threshold_1[user]['TH1'] = alpha*Threshold_1[user]['TH1'] + Threshold_1[user]['ome']
                                Threshold_1[user]['TH2_1'] = alpha*Threshold_1[user]['TH2_1'] - Threshold_1[user]['ome']
                                Threshold_1[user]['TH3_10'] = alpha*Threshold_1[user]['TH3_10'] + Threshold_1[user]['ome']
                                Threshold_1[user]['TH4_101'] = alpha*Threshold_1[user]['TH4_101'] + Threshold_1[user]['ome'] 
                            Threshold_1[user]['TH1'] = np.clip(Threshold_1[user]['TH1'],-Z,Z)
                            Threshold_1[user]['TH2_1'] = np.clip(Threshold_1[user]['TH2_1'],-Z,Z)
                            Threshold_1[user]['TH3_10'] = np.clip(Threshold_1[user]['TH3_10'],-Z,Z)
                            Threshold_1[user]['TH4_101'] = np.clip(Threshold_1[user]['TH4_101'],-Z,Z)
                        else:
                            j4 = '0'
                            j = j1+j2+j3+j4
                            earn = bernoulli.rvs(signal_path_1[user][j][0])
                            signal_path_1[user][j][1] = signal_path_1[user][j][1] + 1 
                            signal_path_1[user][j][2] = signal_path_1[user][j][2] + earn
                            #priv_1[user]['path'] = j
                            # priv_fuzz[user]['ave_earn'][j] = signal_path_1[user][j][2]/signal_path_1[user][j][1]
                            priv_1[user]['ave_earn'][j] = signal_path_1[user][j][2]/signal_path_1[user][j][1]
                            if(earn > 0):
                                Threshold_1[user]['TH1'] = alpha*Threshold_1[user]['TH1'] - Threshold_1[user]['lam'] 
                                Threshold_1[user]['TH2_1'] = alpha*Threshold_1[user]['TH2_1'] + Threshold_1[user]['lam']
                                Threshold_1[user]['TH3_10'] = alpha*Threshold_1[user]['TH3_10'] - Threshold_1[user]['lam']
                                Threshold_1[user]['TH4_101'] = alpha*Threshold_1[user]['TH4_101'] + Threshold_1[user]['lam']
                            else:
                                Threshold_1[user]['TH1'] = alpha*Threshold_1[user]['TH1'] + Threshold_1[user]['ome']
                                Threshold_1[user]['TH2_1'] = alpha*Threshold_1[user]['TH2_1'] - Threshold_1[user]['ome']
                                Threshold_1[user]['TH3_10'] = alpha*Threshold_1[user]['TH3_10'] + Threshold_1[user]['ome']
                                Threshold_1[user]['TH4_101'] = alpha*Threshold_1[user]['TH4_101'] - Threshold_1[user]['ome'] 
                            Threshold_1[user]['TH1'] = np.clip(Threshold_1[user]['TH1'],-Z,Z)
                            Threshold_1[user]['TH2_1'] = np.clip(Threshold_1[user]['TH2_1'],-Z,Z)
                            Threshold_1[user]['TH3_10'] = np.clip(Threshold_1[user]['TH3_10'],-Z,Z)
                            Threshold_1[user]['TH4_101'] = np.clip(Threshold_1[user]['TH4_101'],-Z,Z)
                    else:
                        j3 = '0'
                        if(s4>= Threshold_1[user]['TH4_100']):
                            j4 = '1'
                            j = j1+j2+j3+j4
                            earn = bernoulli.rvs(signal_path_1[user][j][0])
                            signal_path_1[user][j][1] = signal_path_1[user][j][1] + 1 
                            signal_path_1[user][j][2] = signal_path_1[user][j][2] + earn
                            #priv_1[user]['path'] = j
                            # priv_fuzz[user]['ave_earn'][j] = signal_path_1[user][j][2]/signal_path_1[user][j][1]
                            priv_1[user]['ave_earn'][j] = signal_path_1[user][j][2]/signal_path_1[user][j][1]
                            if(earn > 0):
                                Threshold_1[user]['TH1'] = alpha*Threshold_1[user]['TH1'] - Threshold_1[user]['lam'] 
                                Threshold_1[user]['TH2_1'] = alpha*Threshold_1[user]['TH2_1'] + Threshold_1[user]['lam']
                                Threshold_1[user]['TH3_10'] = alpha*Threshold_1[user]['TH3_10'] + Threshold_1[user]['lam']
                                Threshold_1[user]['TH4_100'] = alpha*Threshold_1[user]['TH4_100'] - Threshold_1[user]['lam']
                            else:
                                Threshold_1[user]['TH1'] = alpha*Threshold_1[user]['TH1'] + Threshold_1[user]['ome']
                                Threshold_1[user]['TH2_1'] = alpha*Threshold_1[user]['TH2_1'] - Threshold_1[user]['ome']
                                Threshold_1[user]['TH3_10'] = alpha*Threshold_1[user]['TH3_10'] - Threshold_1[user]['ome']
                                Threshold_1[user]['TH4_100'] = alpha*Threshold_1[user]['TH4_100'] + Threshold_1[user]['ome'] 
                            Threshold_1[user]['TH1'] = np.clip(Threshold_1[user]['TH1'],-Z,Z)
                            Threshold_1[user]['TH2_1'] = np.clip(Threshold_1[user]['TH2_1'],-Z,Z)
                            Threshold_1[user]['TH3_10'] = np.clip(Threshold_1[user]['TH3_10'],-Z,Z)
                            Threshold_1[user]['TH4_100'] = np.clip(Threshold_1[user]['TH4_100'],-Z,Z)
                        else:
                            j4 = '0'
                            j = j1+j2+j3+j4
                            earn = bernoulli.rvs(signal_path_1[user][j][0])
                            signal_path_1[user][j][1] = signal_path_1[user][j][1] + 1 
                            signal_path_1[user][j][2] = signal_path_1[user][j][2] + earn
                            #priv_1[user]['path'] = j
                            # priv_fuzz[user]['ave_earn'][j] = signal_path_1[user][j][2]/signal_path_1[user][j][1]
                            priv_1[user]['ave_earn'][j] = signal_path_1[user][j][2]/signal_path_1[user][j][1]
                            if(earn > 0):
                                Threshold_1[user]['TH1'] = alpha*Threshold_1[user]['TH1'] - Threshold_1[user]['lam'] 
                                Threshold_1[user]['TH2_1'] = alpha*Threshold_1[user]['TH2_1'] + Threshold_1[user]['lam']
                                Threshold_1[user]['TH3_10'] = alpha*Threshold_1[user]['TH3_10'] + Threshold_1[user]['lam']
                                Threshold_1[user]['TH4_100'] = alpha*Threshold_1[user]['TH4_100'] + Threshold_1[user]['lam']
                            else:
                                Threshold_1[user]['TH1'] = alpha*Threshold_1[user]['TH1'] + Threshold_1[user]['ome']
                                Threshold_1[user]['TH2_1'] = alpha*Threshold_1[user]['TH2_1'] - Threshold_1[user]['ome']
                                Threshold_1[user]['TH3_10'] = alpha*Threshold_1[user]['TH3_10'] - Threshold_1[user]['ome']
                                Threshold_1[user]['TH4_100'] = alpha*Threshold_1[user]['TH4_100'] - Threshold_1[user]['ome'] 
                            Threshold_1[user]['TH1'] = np.clip(Threshold_1[user]['TH1'],-Z,Z)
                            Threshold_1[user]['TH2_1'] = np.clip(Threshold_1[user]['TH2_1'],-Z,Z)
                            Threshold_1[user]['TH3_10'] = np.clip(Threshold_1[user]['TH3_10'],-Z,Z)
                            Threshold_1[user]['TH4_100'] = np.clip(Threshold_1[user]['TH4_100'],-Z,Z)
            else:
                j1 = '0'
                if(s2>= Threshold_1[user]['TH2_0']):
                    j2 = '1'
                    if(s3>= Threshold_1[user]['TH3_01']):
                        j3 = '1'
                        if(s4>= Threshold_1[user]['TH4_011']):
                            j4 = '1'
                            j = j1+j2+j3+j4
                            earn = bernoulli.rvs(signal_path_1[user][j][0])
                            signal_path_1[user][j][1] = signal_path_1[user][j][1] + 1 
                            signal_path_1[user][j][2] = signal_path_1[user][j][2] + earn
                            #priv_1[user]['path'] = j
                            # priv_fuzz[user]['ave_earn'][j] = signal_path_1[user][j][2]/signal_path_1[user][j][1]
                            priv_1[user]['ave_earn'][j] = signal_path_1[user][j][2]/signal_path_1[user][j][1]
                            if(earn > 0):
                                Threshold_1[user]['TH1'] = alpha*Threshold_1[user]['TH1'] + Threshold_1[user]['lam'] 
                                Threshold_1[user]['TH2_0'] = alpha*Threshold_1[user]['TH2_0'] - Threshold_1[user]['lam']
                                Threshold_1[user]['TH3_01'] = alpha*Threshold_1[user]['TH3_01'] - Threshold_1[user]['lam']
                                Threshold_1[user]['TH4_011'] = alpha*Threshold_1[user]['TH4_011'] - Threshold_1[user]['lam']
                            else:
                                Threshold_1[user]['TH1'] = alpha*Threshold_1[user]['TH1'] - Threshold_1[user]['ome']
                                Threshold_1[user]['TH2_0'] = alpha*Threshold_1[user]['TH2_0'] + Threshold_1[user]['ome']
                                Threshold_1[user]['TH3_01'] = alpha*Threshold_1[user]['TH3_01'] + Threshold_1[user]['ome']
                                Threshold_1[user]['TH4_011'] = alpha*Threshold_1[user]['TH4_011'] + Threshold_1[user]['ome'] 
                            Threshold_1[user]['TH1'] = np.clip(Threshold_1[user]['TH1'],-Z,Z)
                            Threshold_1[user]['TH2_0'] = np.clip(Threshold_1[user]['TH2_0'],-Z,Z)
                            Threshold_1[user]['TH3_01'] = np.clip(Threshold_1[user]['TH3_01'],-Z,Z)
                            Threshold_1[user]['TH4_011'] = np.clip(Threshold_1[user]['TH4_011'],-Z,Z)
                        else:
                            j4 = '0'
                            j = j1+j2+j3+j4
                            earn = bernoulli.rvs(signal_path_1[user][j][0])
                            signal_path_1[user][j][1] = signal_path_1[user][j][1] + 1 
                            signal_path_1[user][j][2] = signal_path_1[user][j][2] + earn
                            #priv_1[user]['path'] = j
                            # priv_fuzz[user]['ave_earn'][j] = signal_path_1[user][j][2]/signal_path_1[user][j][1]
                            priv_1[user]['ave_earn'][j] = signal_path_1[user][j][2]/signal_path_1[user][j][1]
                            if(earn > 0):
                                Threshold_1[user]['TH1'] = alpha*Threshold_1[user]['TH1'] + Threshold_1[user]['lam'] 
                                Threshold_1[user]['TH2_0'] = alpha*Threshold_1[user]['TH2_0'] - Threshold_1[user]['lam']
                                Threshold_1[user]['TH3_01'] = alpha*Threshold_1[user]['TH3_01'] - Threshold_1[user]['lam']
                                Threshold_1[user]['TH4_011'] = alpha*Threshold_1[user]['TH4_011'] + Threshold_1[user]['lam']
                            else:
                                Threshold_1[user]['TH1'] = alpha*Threshold_1[user]['TH1'] - Threshold_1[user]['ome']
                                Threshold_1[user]['TH2_0'] = alpha*Threshold_1[user]['TH2_0'] + Threshold_1[user]['ome']
                                Threshold_1[user]['TH3_01'] = alpha*Threshold_1[user]['TH3_01'] + Threshold_1[user]['ome']
                                Threshold_1[user]['TH4_011'] = alpha*Threshold_1[user]['TH4_011'] - Threshold_1[user]['ome'] 
                            Threshold_1[user]['TH1'] = np.clip(Threshold_1[user]['TH1'],-Z,Z)
                            Threshold_1[user]['TH2_0'] = np.clip(Threshold_1[user]['TH2_0'],-Z,Z)
                            Threshold_1[user]['TH3_01'] = np.clip(Threshold_1[user]['TH3_01'],-Z,Z)
                            Threshold_1[user]['TH4_011'] = np.clip(Threshold_1[user]['TH4_011'],-Z,Z)
                    else:
                        j3 = '0'
                        if(s4>= Threshold_1[user]['TH4_010']):
                            j4 = '1'
                            j = j1+j2+j3+j4
                            earn = bernoulli.rvs(signal_path_1[user][j][0])
                            signal_path_1[user][j][1] = signal_path_1[user][j][1] + 1 
                            signal_path_1[user][j][2] = signal_path_1[user][j][2] + earn
                            #priv_1[user]['path'] = j
                            # priv_fuzz[user]['ave_earn'][j] = signal_path_1[user][j][2]/signal_path_1[user][j][1]
                            priv_1[user]['ave_earn'][j] = signal_path_1[user][j][2]/signal_path_1[user][j][1]
                            if(earn > 0):
                                Threshold_1[user]['TH1'] = alpha*Threshold_1[user]['TH1'] + Threshold_1[user]['lam'] 
                                Threshold_1[user]['TH2_0'] = alpha*Threshold_1[user]['TH2_0'] - Threshold_1[user]['lam']
                                Threshold_1[user]['TH3_01'] = alpha*Threshold_1[user]['TH3_01'] + Threshold_1[user]['lam']
                                Threshold_1[user]['TH4_010'] = alpha*Threshold_1[user]['TH4_010'] - Threshold_1[user]['lam']
                            else:
                                Threshold_1[user]['TH1'] = alpha*Threshold_1[user]['TH1'] - Threshold_1[user]['ome']
                                Threshold_1[user]['TH2_0'] = alpha*Threshold_1[user]['TH2_0'] + Threshold_1[user]['ome']
                                Threshold_1[user]['TH3_01'] = alpha*Threshold_1[user]['TH3_01'] - Threshold_1[user]['ome']
                                Threshold_1[user]['TH4_010'] = alpha*Threshold_1[user]['TH4_010'] + Threshold_1[user]['ome']
                            Threshold_1[user]['TH1'] = np.clip(Threshold_1[user]['TH1'],-Z,Z)
                            Threshold_1[user]['TH2_0'] = np.clip(Threshold_1[user]['TH2_0'],-Z,Z)
                            Threshold_1[user]['TH3_01'] = np.clip(Threshold_1[user]['TH3_01'],-Z,Z)
                            Threshold_1[user]['TH4_010'] = np.clip(Threshold_1[user]['TH4_010'],-Z,Z)
                        else:
                            j4 = '0'
                            j = j1+j2+j3+j4
                            earn = bernoulli.rvs(signal_path_1[user][j][0])
                            signal_path_1[user][j][1] = signal_path_1[user][j][1] + 1 
                            signal_path_1[user][j][2] = signal_path_1[user][j][2] + earn
                            #priv_1[user]['path'] = j
                            # priv_fuzz[user]['ave_earn'][j] = signal_path_1[user][j][2]/signal_path_1[user][j][1]
                            priv_1[user]['ave_earn'][j] = signal_path_1[user][j][2]/signal_path_1[user][j][1]
                            if(earn > 0):
                                Threshold_1[user]['TH1'] = alpha*Threshold_1[user]['TH1'] + Threshold_1[user]['lam'] 
                                Threshold_1[user]['TH2_0'] = alpha*Threshold_1[user]['TH2_0'] - Threshold_1[user]['lam']
                                Threshold_1[user]['TH3_01'] = alpha*Threshold_1[user]['TH3_01'] + Threshold_1[user]['lam']
                                Threshold_1[user]['TH4_010'] = alpha*Threshold_1[user]['TH4_010'] + Threshold_1[user]['lam']
                            else:
                                Threshold_1[user]['TH1'] = alpha*Threshold_1[user]['TH1'] - Threshold_1[user]['ome']
                                Threshold_1[user]['TH2_0'] = alpha*Threshold_1[user]['TH2_0'] + Threshold_1[user]['ome']
                                Threshold_1[user]['TH3_01'] = alpha*Threshold_1[user]['TH3_01'] - Threshold_1[user]['ome']
                                Threshold_1[user]['TH4_010'] = alpha*Threshold_1[user]['TH4_010'] - Threshold_1[user]['ome']
                            Threshold_1[user]['TH1'] = np.clip(Threshold_1[user]['TH1'],-Z,Z)
                            Threshold_1[user]['TH2_0'] = np.clip(Threshold_1[user]['TH2_0'],-Z,Z)
                            Threshold_1[user]['TH3_01'] = np.clip(Threshold_1[user]['TH3_01'],-Z,Z)
                            Threshold_1[user]['TH4_010'] = np.clip(Threshold_1[user]['TH4_010'],-Z,Z)
                else:
                    j2 = '0'
                    if(s3>= Threshold_1[user]['TH3_00']):
                        j3 = '1'
                        if(s4>= Threshold_1[user]['TH4_001']):
                            j4 = '1'
                            j = j1+j2+j3+j4
                            earn = bernoulli.rvs(signal_path_1[user][j][0])
                            signal_path_1[user][j][1] = signal_path_1[user][j][1] + 1 
                            signal_path_1[user][j][2] = signal_path_1[user][j][2] + earn
                            #priv_1[user]['path'] = j
                            # priv_fuzz[user]['ave_earn'][j] = signal_path_1[user][j][2]/signal_path_1[user][j][1]
                            priv_1[user]['ave_earn'][j] = signal_path_1[user][j][2]/signal_path_1[user][j][1]
                            if(earn > 0):
                                Threshold_1[user]['TH1'] = alpha*Threshold_1[user]['TH1'] + Threshold_1[user]['lam'] 
                                Threshold_1[user]['TH2_0'] = alpha*Threshold_1[user]['TH2_0'] + Threshold_1[user]['lam']
                                Threshold_1[user]['TH3_00'] = alpha*Threshold_1[user]['TH3_00'] - Threshold_1[user]['lam']
                                Threshold_1[user]['TH4_001'] = alpha*Threshold_1[user]['TH4_001'] - Threshold_1[user]['lam']
                            else:
                                Threshold_1[user]['TH1'] = alpha*Threshold_1[user]['TH1'] - Threshold_1[user]['ome']
                                Threshold_1[user]['TH2_0'] = alpha*Threshold_1[user]['TH2_0'] - Threshold_1[user]['ome']
                                Threshold_1[user]['TH3_00'] = alpha*Threshold_1[user]['TH3_00'] + Threshold_1[user]['ome']
                                Threshold_1[user]['TH4_001'] = alpha*Threshold_1[user]['TH4_001'] + Threshold_1[user]['ome']
                            Threshold_1[user]['TH1'] = np.clip(Threshold_1[user]['TH1'],-Z,Z)
                            Threshold_1[user]['TH2_0'] = np.clip(Threshold_1[user]['TH2_0'],-Z,Z)
                            Threshold_1[user]['TH3_00'] = np.clip(Threshold_1[user]['TH3_00'],-Z,Z)
                            Threshold_1[user]['TH4_001'] = np.clip(Threshold_1[user]['TH4_001'],-Z,Z)
                        else:
                            j4 = '0'
                            j = j1+j2+j3+j4
                            earn = bernoulli.rvs(signal_path_1[user][j][0])
                            signal_path_1[user][j][1] = signal_path_1[user][j][1] + 1 
                            signal_path_1[user][j][2] = signal_path_1[user][j][2] + earn
                            #priv_1[user]['path'] = j
                            # priv_fuzz[user]['ave_earn'][j] = signal_path_1[user][j][2]/signal_path_1[user][j][1]
                            priv_1[user]['ave_earn'][j] = signal_path_1[user][j][2]/signal_path_1[user][j][1]
                            if(earn > 0):
                                Threshold_1[user]['TH1'] = alpha*Threshold_1[user]['TH1'] + Threshold_1[user]['lam'] 
                                Threshold_1[user]['TH2_0'] = alpha*Threshold_1[user]['TH2_0'] + Threshold_1[user]['lam']
                                Threshold_1[user]['TH3_00'] = alpha*Threshold_1[user]['TH3_00'] - Threshold_1[user]['lam']
                                Threshold_1[user]['TH4_001'] = alpha*Threshold_1[user]['TH4_001'] + Threshold_1[user]['lam']
                            else:
                                Threshold_1[user]['TH1'] = alpha*Threshold_1[user]['TH1'] - Threshold_1[user]['ome']
                                Threshold_1[user]['TH2_0'] = alpha*Threshold_1[user]['TH2_0'] - Threshold_1[user]['ome']
                                Threshold_1[user]['TH3_00'] = alpha*Threshold_1[user]['TH3_00'] + Threshold_1[user]['ome']
                                Threshold_1[user]['TH4_001'] = alpha*Threshold_1[user]['TH4_001'] - Threshold_1[user]['ome']
                            Threshold_1[user]['TH1'] = np.clip(Threshold_1[user]['TH1'],-Z,Z)
                            Threshold_1[user]['TH2_0'] = np.clip(Threshold_1[user]['TH2_0'],-Z,Z)
                            Threshold_1[user]['TH3_00'] = np.clip(Threshold_1[user]['TH3_00'],-Z,Z)
                            Threshold_1[user]['TH4_001'] = np.clip(Threshold_1[user]['TH4_001'],-Z,Z)
                    else:
                        j3 = '0'
                        if(s4>= Threshold_1[user]['TH4_000']):
                            j4 = '1'
                            j = j1+j2+j3+j4
                            earn = bernoulli.rvs(signal_path_1[user][j][0])
                            signal_path_1[user][j][1] = signal_path_1[user][j][1] + 1 
                            signal_path_1[user][j][2] = signal_path_1[user][j][2] + earn
                            #priv_1[user]['path'] = j
                            # priv_fuzz[user]['ave_earn'][j] = signal_path_1[user][j][2]/signal_path_1[user][j][1]
                            priv_1[user]['ave_earn'][j] = signal_path_1[user][j][2]/signal_path_1[user][j][1]
                            if(earn > 0):
                                Threshold_1[user]['TH1'] = alpha*Threshold_1[user]['TH1'] + Threshold_1[user]['lam'] 
                                Threshold_1[user]['TH2_0'] = alpha*Threshold_1[user]['TH2_0'] + Threshold_1[user]['lam']
                                Threshold_1[user]['TH3_00'] = alpha*Threshold_1[user]['TH3_00'] + Threshold_1[user]['lam']
                                Threshold_1[user]['TH4_000'] = alpha*Threshold_1[user]['TH4_000'] - Threshold_1[user]['lam']
                            else:
                                Threshold_1[user]['TH1'] = alpha*Threshold_1[user]['TH1'] - Threshold_1[user]['ome']
                                Threshold_1[user]['TH2_0'] = alpha*Threshold_1[user]['TH2_0'] - Threshold_1[user]['ome']
                                Threshold_1[user]['TH3_00'] = alpha*Threshold_1[user]['TH3_00'] - Threshold_1[user]['ome']
                                Threshold_1[user]['TH4_000'] = alpha*Threshold_1[user]['TH4_000'] + Threshold_1[user]['ome']
                            Threshold_1[user]['TH1'] = np.clip(Threshold_1[user]['TH1'],-Z,Z)
                            Threshold_1[user]['TH2_0'] = np.clip(Threshold_1[user]['TH2_0'],-Z,Z)
                            Threshold_1[user]['TH3_00'] = np.clip(Threshold_1[user]['TH3_00'],-Z,Z)
                            Threshold_1[user]['TH4_000'] = np.clip(Threshold_1[user]['TH4_000'],-Z,Z)
                        else:
                            j4 = '0'
                            j = j1+j2+j3+j4
                            earn = bernoulli.rvs(signal_path_1[user][j][0])
                            signal_path_1[user][j][1] = signal_path_1[user][j][1] + 1 
                            signal_path_1[user][j][2] = signal_path_1[user][j][2] + earn
                            #priv_1[user]['path'] = j
                            # priv_fuzz[user]['ave_earn'][j] = signal_path_1[user][j][2]/signal_path_1[user][j][1]
                            priv_1[user]['ave_earn'][j] = signal_path_1[user][j][2]/signal_path_1[user][j][1]
                            if(earn > 0):
                                Threshold_1[user]['TH1'] = alpha*Threshold_1[user]['TH1'] + Threshold_1[user]['lam'] 
                                Threshold_1[user]['TH2_0'] = alpha*Threshold_1[user]['TH2_0'] + Threshold_1[user]['lam']
                                Threshold_1[user]['TH3_00'] = alpha*Threshold_1[user]['TH3_00'] + Threshold_1[user]['lam']
                                Threshold_1[user]['TH4_000'] = alpha*Threshold_1[user]['TH4_000'] + Threshold_1[user]['lam']
                            else:
                                Threshold_1[user]['TH1'] = alpha*Threshold_1[user]['TH1'] - Threshold_1[user]['ome']
                                Threshold_1[user]['TH2_0'] = alpha*Threshold_1[user]['TH2_0'] - Threshold_1[user]['ome']
                                Threshold_1[user]['TH3_00'] = alpha*Threshold_1[user]['TH3_00'] - Threshold_1[user]['ome']
                                Threshold_1[user]['TH4_000'] = alpha*Threshold_1[user]['TH4_000'] - Threshold_1[user]['ome']
                            Threshold_1[user]['TH1'] = np.clip(Threshold_1[user]['TH1'],-Z,Z)
                            Threshold_1[user]['TH2_0'] = np.clip(Threshold_1[user]['TH2_0'],-Z,Z)
                            Threshold_1[user]['TH3_00'] = np.clip(Threshold_1[user]['TH3_00'],-Z,Z)
                            Threshold_1[user]['TH4_000'] = np.clip(Threshold_1[user]['TH4_000'],-Z,Z)

            seq = []
            prefs = sorted(priv_1[user]['ave_earn'].items(),key = lambda x:x[1],reverse= True)
            for pref in prefs:
                seq.append(pref[0])
            priv_1[user]['pref'] = seq
            

            if(chaos_list[time]>= Threshold_2[user]['TH1']):
                j1 = '1'       
                if(chaos_list[time+1]>= Threshold_2[user]['TH2_1']):
                    j2 = '1'
                    if(chaos_list[time+2]>= Threshold_2[user]['TH3_11']):
                        j3 = '1'
                        if(chaos_list[time+3]>= Threshold_2[user]['TH4_111']):
                            j4 = '1'
                            j = j1+j2+j3+j4
                            earn = bernoulli.rvs(signal_path_2[user][j][0])
                            signal_path_2[user][j][1] = signal_path_2[user][j][1] + 1 
                            signal_path_2[user][j][2] = signal_path_2[user][j][2] + earn
                            #priv_1[user]['path'] = j
                            priv_2[user]['ave_earn'][j] = signal_path_2[user][j][2]/signal_path_2[user][j][1]
                            # priv_fuzz[user]['ave_earn'][j] = signal_path_2[user][j][2]/signal_path_2[user][j][1]
                            if(earn > 0):
                                Threshold_2[user]['TH1'] = alpha*Threshold_2[user]['TH1'] - Threshold_2[user]['lam'] 
                                Threshold_2[user]['TH2_1'] = alpha*Threshold_2[user]['TH2_1'] - Threshold_2[user]['lam']
                                Threshold_2[user]['TH3_11'] = alpha*Threshold_2[user]['TH3_11'] - Threshold_2[user]['lam']
                                Threshold_2[user]['TH4_111'] = alpha*Threshold_2[user]['TH4_111'] - Threshold_2[user]['lam']
                            else:
                                Threshold_2[user]['TH1'] = alpha*Threshold_2[user]['TH1'] + Threshold_2[user]['ome']
                                Threshold_2[user]['TH2_1'] = alpha*Threshold_2[user]['TH2_1'] + Threshold_2[user]['ome']
                                Threshold_2[user]['TH3_11'] = alpha*Threshold_2[user]['TH3_11'] + Threshold_2[user]['ome']
                                Threshold_2[user]['TH4_111'] = alpha*Threshold_2[user]['TH4_111'] + Threshold_2[user]['ome'] 
                            Threshold_2[user]['TH1'] = np.clip(Threshold_2[user]['TH1'],-Z,Z)
                            Threshold_2[user]['TH2_1'] = np.clip(Threshold_2[user]['TH2_1'],-Z,Z)
                            Threshold_2[user]['TH3_11'] = np.clip(Threshold_2[user]['TH3_11'],-Z,Z)
                            Threshold_2[user]['TH4_111'] = np.clip(Threshold_2[user]['TH4_111'],-Z,Z)
                        else:
                            j4 = '0'
                            j = j1+j2+j3+j4
                            earn = bernoulli.rvs(signal_path_2[user][j][0])
                            signal_path_2[user][j][1] = signal_path_2[user][j][1] + 1 
                            signal_path_2[user][j][2] = signal_path_2[user][j][2] + earn
                            #priv_2[user]['path'] = j
                            # priv_fuzz[user]['ave_earn'][j] = signal_path_2[user][j][2]/signal_path_2[user][j][1]
                            priv_2[user]['ave_earn'][j] = signal_path_2[user][j][2]/signal_path_2[user][j][1]
                            if(earn > 0):
                                Threshold_2[user]['TH1'] = alpha*Threshold_2[user]['TH1'] - Threshold_2[user]['lam'] 
                                Threshold_2[user]['TH2_1'] = alpha*Threshold_2[user]['TH2_1'] - Threshold_2[user]['lam']
                                Threshold_2[user]['TH3_11'] = alpha*Threshold_2[user]['TH3_11'] - Threshold_2[user]['lam']
                                Threshold_2[user]['TH4_111'] = alpha*Threshold_2[user]['TH4_111'] + Threshold_2[user]['lam']
                            else:
                                Threshold_2[user]['TH1'] = alpha*Threshold_2[user]['TH1'] + Threshold_2[user]['ome']
                                Threshold_2[user]['TH2_1'] = alpha*Threshold_2[user]['TH2_1'] + Threshold_2[user]['ome']
                                Threshold_2[user]['TH3_11'] = alpha*Threshold_2[user]['TH3_11'] + Threshold_2[user]['ome']
                                Threshold_2[user]['TH4_111'] = alpha*Threshold_2[user]['TH4_111'] - Threshold_2[user]['ome'] 
                            Threshold_2[user]['TH1'] = np.clip(Threshold_2[user]['TH1'],-Z,Z)
                            Threshold_2[user]['TH2_1'] = np.clip(Threshold_2[user]['TH2_1'],-Z,Z)
                            Threshold_2[user]['TH3_11'] = np.clip(Threshold_2[user]['TH3_11'],-Z,Z)
                            Threshold_2[user]['TH4_111'] = np.clip(Threshold_2[user]['TH4_111'],-Z,Z)
                    else:
                        j3 = '0'
                        if(chaos_list[time+3]>= Threshold_2[user]['TH4_110']):
                            j4 = '1'
                            j = j1+j2+j3+j4
                            earn = bernoulli.rvs(signal_path_2[user][j][0])
                            signal_path_2[user][j][1] = signal_path_2[user][j][1] + 1 
                            signal_path_2[user][j][2] = signal_path_2[user][j][2] + earn
                            #priv_2[user]['path'] = j
                            # priv_fuzz[user]['ave_earn'][j] = signal_path_2[user][j][2]/signal_path_2[user][j][1]
                            priv_2[user]['ave_earn'][j] = signal_path_2[user][j][2]/signal_path_2[user][j][1]
                            if(earn > 0):
                                Threshold_2[user]['TH1'] = alpha*Threshold_2[user]['TH1'] - Threshold_2[user]['lam'] 
                                Threshold_2[user]['TH2_1'] = alpha*Threshold_2[user]['TH2_1'] - Threshold_2[user]['lam']
                                Threshold_2[user]['TH3_11'] = alpha*Threshold_2[user]['TH3_11'] + Threshold_2[user]['lam']
                                Threshold_2[user]['TH4_110'] = alpha*Threshold_2[user]['TH4_110'] - Threshold_2[user]['lam']
                            else:
                                Threshold_2[user]['TH1'] = alpha*Threshold_2[user]['TH1'] + Threshold_2[user]['ome']
                                Threshold_2[user]['TH2_1'] = alpha*Threshold_2[user]['TH2_1'] + Threshold_2[user]['ome']
                                Threshold_2[user]['TH3_11'] = alpha*Threshold_2[user]['TH3_11'] - Threshold_2[user]['ome']
                                Threshold_2[user]['TH4_110'] = alpha*Threshold_2[user]['TH4_110'] + Threshold_2[user]['ome'] 
                            Threshold_2[user]['TH1'] = np.clip(Threshold_2[user]['TH1'],-Z,Z)
                            Threshold_2[user]['TH2_1'] = np.clip(Threshold_2[user]['TH2_1'],-Z,Z)
                            Threshold_2[user]['TH3_11'] = np.clip(Threshold_2[user]['TH3_11'],-Z,Z)
                            Threshold_2[user]['TH4_110'] = np.clip(Threshold_2[user]['TH4_110'],-Z,Z)
                        else:
                            j4 = '0'
                            j = j1+j2+j3+j4
                            earn = bernoulli.rvs(signal_path_2[user][j][0])
                            signal_path_2[user][j][1] = signal_path_2[user][j][1] + 1 
                            signal_path_2[user][j][2] = signal_path_2[user][j][2] + earn
                            #priv_2[user]['path'] = j
                            # priv_fuzz[user]['ave_earn'][j] = signal_path_2[user][j][2]/signal_path_2[user][j][1]
                            priv_2[user]['ave_earn'][j] = signal_path_2[user][j][2]/signal_path_2[user][j][1]
                            if(earn > 0):
                                Threshold_2[user]['TH1'] = alpha*Threshold_2[user]['TH1'] - Threshold_2[user]['lam'] 
                                Threshold_2[user]['TH2_1'] = alpha*Threshold_2[user]['TH2_1'] - Threshold_2[user]['lam']
                                Threshold_2[user]['TH3_11'] = alpha*Threshold_2[user]['TH3_11'] + Threshold_2[user]['lam']
                                Threshold_2[user]['TH4_110'] = alpha*Threshold_2[user]['TH4_110'] + Threshold_2[user]['lam']
                            else:
                                Threshold_2[user]['TH1'] = alpha*Threshold_2[user]['TH1'] + Threshold_2[user]['ome']
                                Threshold_2[user]['TH2_1'] = alpha*Threshold_2[user]['TH2_1'] + Threshold_2[user]['ome']
                                Threshold_2[user]['TH3_11'] = alpha*Threshold_2[user]['TH3_11'] - Threshold_2[user]['ome']
                                Threshold_2[user]['TH4_110'] = alpha*Threshold_2[user]['TH4_110'] - Threshold_2[user]['ome'] 
                            Threshold_2[user]['TH1'] = np.clip(Threshold_2[user]['TH1'],-Z,Z)
                            Threshold_2[user]['TH2_1'] = np.clip(Threshold_2[user]['TH2_1'],-Z,Z)
                            Threshold_2[user]['TH3_11'] = np.clip(Threshold_2[user]['TH3_11'],-Z,Z)
                            Threshold_2[user]['TH4_110'] = np.clip(Threshold_2[user]['TH4_110'],-Z,Z)                     
                else:
                    j2 = '0'
                    if(chaos_list[time+2]>= Threshold_2[user]['TH3_10']):
                        j3 = '1'
                        if(chaos_list[time+3]>= Threshold_2[user]['TH4_101']):
                            j4 = '1'
                            j = j1+j2+j3+j4
                            earn = bernoulli.rvs(signal_path_2[user][j][0])
                            signal_path_2[user][j][1] = signal_path_2[user][j][1] + 1 
                            signal_path_2[user][j][2] = signal_path_2[user][j][2] + earn
                            #priv_2[user]['path'] = j
                            # priv_fuzz[user]['ave_earn'][j] = signal_path_2[user][j][2]/signal_path_2[user][j][1]
                            priv_2[user]['ave_earn'][j] = signal_path_2[user][j][2]/signal_path_2[user][j][1]
                            if(earn > 0):
                                Threshold_2[user]['TH1'] = alpha*Threshold_2[user]['TH1'] - Threshold_2[user]['lam'] 
                                Threshold_2[user]['TH2_1'] = alpha*Threshold_2[user]['TH2_1'] + Threshold_2[user]['lam']
                                Threshold_2[user]['TH3_10'] = alpha*Threshold_2[user]['TH3_10'] - Threshold_2[user]['lam']
                                Threshold_2[user]['TH4_101'] = alpha*Threshold_2[user]['TH4_101'] - Threshold_2[user]['lam']
                            else:
                                Threshold_2[user]['TH1'] = alpha*Threshold_2[user]['TH1'] + Threshold_2[user]['ome']
                                Threshold_2[user]['TH2_1'] = alpha*Threshold_2[user]['TH2_1'] - Threshold_2[user]['ome']
                                Threshold_2[user]['TH3_10'] = alpha*Threshold_2[user]['TH3_10'] + Threshold_2[user]['ome']
                                Threshold_2[user]['TH4_101'] = alpha*Threshold_2[user]['TH4_101'] + Threshold_2[user]['ome'] 
                            Threshold_2[user]['TH1'] = np.clip(Threshold_2[user]['TH1'],-Z,Z)
                            Threshold_2[user]['TH2_1'] = np.clip(Threshold_2[user]['TH2_1'],-Z,Z)
                            Threshold_2[user]['TH3_10'] = np.clip(Threshold_2[user]['TH3_10'],-Z,Z)
                            Threshold_2[user]['TH4_101'] = np.clip(Threshold_2[user]['TH4_101'],-Z,Z)
                        else:
                            j4 = '0'
                            j = j1+j2+j3+j4
                            earn = bernoulli.rvs(signal_path_2[user][j][0])
                            signal_path_2[user][j][1] = signal_path_2[user][j][1] + 1 
                            signal_path_2[user][j][2] = signal_path_2[user][j][2] + earn
                            #priv_2[user]['path'] = j
                            # priv_fuzz[user]['ave_earn'][j] = signal_path_2[user][j][2]/signal_path_2[user][j][1]
                            priv_2[user]['ave_earn'][j] = signal_path_2[user][j][2]/signal_path_2[user][j][1]
                            if(earn > 0):
                                Threshold_2[user]['TH1'] = alpha*Threshold_2[user]['TH1'] - Threshold_2[user]['lam'] 
                                Threshold_2[user]['TH2_1'] = alpha*Threshold_2[user]['TH2_1'] + Threshold_2[user]['lam']
                                Threshold_2[user]['TH3_10'] = alpha*Threshold_2[user]['TH3_10'] - Threshold_2[user]['lam']
                                Threshold_2[user]['TH4_101'] = alpha*Threshold_2[user]['TH4_101'] + Threshold_2[user]['lam']
                            else:
                                Threshold_2[user]['TH1'] = alpha*Threshold_2[user]['TH1'] + Threshold_2[user]['ome']
                                Threshold_2[user]['TH2_1'] = alpha*Threshold_2[user]['TH2_1'] - Threshold_2[user]['ome']
                                Threshold_2[user]['TH3_10'] = alpha*Threshold_2[user]['TH3_10'] + Threshold_2[user]['ome']
                                Threshold_2[user]['TH4_101'] = alpha*Threshold_2[user]['TH4_101'] - Threshold_2[user]['ome'] 
                            Threshold_2[user]['TH1'] = np.clip(Threshold_2[user]['TH1'],-Z,Z)
                            Threshold_2[user]['TH2_1'] = np.clip(Threshold_2[user]['TH2_1'],-Z,Z)
                            Threshold_2[user]['TH3_10'] = np.clip(Threshold_2[user]['TH3_10'],-Z,Z)
                            Threshold_2[user]['TH4_101'] = np.clip(Threshold_2[user]['TH4_101'],-Z,Z)
                    else:
                        j3 = '0'
                        if(chaos_list[time+3]>= Threshold_2[user]['TH4_100']):
                            j4 = '1'
                            j = j1+j2+j3+j4
                            earn = bernoulli.rvs(signal_path_2[user][j][0])
                            signal_path_2[user][j][1] = signal_path_2[user][j][1] + 1 
                            signal_path_2[user][j][2] = signal_path_2[user][j][2] + earn
                            #priv_2[user]['path'] = j
                            # priv_fuzz[user]['ave_earn'][j] = signal_path_2[user][j][2]/signal_path_2[user][j][1]
                            priv_2[user]['ave_earn'][j] = signal_path_2[user][j][2]/signal_path_2[user][j][1]
                            if(earn > 0):
                                Threshold_2[user]['TH1'] = alpha*Threshold_2[user]['TH1'] - Threshold_2[user]['lam'] 
                                Threshold_2[user]['TH2_1'] = alpha*Threshold_2[user]['TH2_1'] + Threshold_2[user]['lam']
                                Threshold_2[user]['TH3_10'] = alpha*Threshold_2[user]['TH3_10'] + Threshold_2[user]['lam']
                                Threshold_2[user]['TH4_100'] = alpha*Threshold_2[user]['TH4_100'] - Threshold_2[user]['lam']
                            else:
                                Threshold_2[user]['TH1'] = alpha*Threshold_2[user]['TH1'] + Threshold_2[user]['ome']
                                Threshold_2[user]['TH2_1'] = alpha*Threshold_2[user]['TH2_1'] - Threshold_2[user]['ome']
                                Threshold_2[user]['TH3_10'] = alpha*Threshold_2[user]['TH3_10'] - Threshold_2[user]['ome']
                                Threshold_2[user]['TH4_100'] = alpha*Threshold_2[user]['TH4_100'] + Threshold_2[user]['ome'] 
                            Threshold_2[user]['TH1'] = np.clip(Threshold_2[user]['TH1'],-Z,Z)
                            Threshold_2[user]['TH2_1'] = np.clip(Threshold_2[user]['TH2_1'],-Z,Z)
                            Threshold_2[user]['TH3_10'] = np.clip(Threshold_2[user]['TH3_10'],-Z,Z)
                            Threshold_2[user]['TH4_100'] = np.clip(Threshold_2[user]['TH4_100'],-Z,Z)
                        else:
                            j4 = '0'
                            j = j1+j2+j3+j4
                            earn = bernoulli.rvs(signal_path_2[user][j][0])
                            signal_path_2[user][j][1] = signal_path_2[user][j][1] + 1 
                            signal_path_2[user][j][2] = signal_path_2[user][j][2] + earn
                            #priv_2[user]['path'] = j
                            # priv_fuzz[user]['ave_earn'][j] = signal_path_2[user][j][2]/signal_path_2[user][j][1]
                            priv_2[user]['ave_earn'][j] = signal_path_2[user][j][2]/signal_path_2[user][j][1]
                            if(earn > 0):
                                Threshold_2[user]['TH1'] = alpha*Threshold_2[user]['TH1'] - Threshold_2[user]['lam'] 
                                Threshold_2[user]['TH2_1'] = alpha*Threshold_2[user]['TH2_1'] + Threshold_2[user]['lam']
                                Threshold_2[user]['TH3_10'] = alpha*Threshold_2[user]['TH3_10'] + Threshold_2[user]['lam']
                                Threshold_2[user]['TH4_100'] = alpha*Threshold_2[user]['TH4_100'] + Threshold_2[user]['lam']
                            else:
                                Threshold_2[user]['TH1'] = alpha*Threshold_2[user]['TH1'] + Threshold_2[user]['ome']
                                Threshold_2[user]['TH2_1'] = alpha*Threshold_2[user]['TH2_1'] - Threshold_2[user]['ome']
                                Threshold_2[user]['TH3_10'] = alpha*Threshold_2[user]['TH3_10'] - Threshold_2[user]['ome']
                                Threshold_2[user]['TH4_100'] = alpha*Threshold_2[user]['TH4_100'] - Threshold_2[user]['ome'] 
                            Threshold_2[user]['TH1'] = np.clip(Threshold_2[user]['TH1'],-Z,Z)
                            Threshold_2[user]['TH2_1'] = np.clip(Threshold_2[user]['TH2_1'],-Z,Z)
                            Threshold_2[user]['TH3_10'] = np.clip(Threshold_2[user]['TH3_10'],-Z,Z)
                            Threshold_2[user]['TH4_100'] = np.clip(Threshold_2[user]['TH4_100'],-Z,Z)
            else:
                j1 = '0'
                if(chaos_list[time+1]>= Threshold_2[user]['TH2_0']):
                    j2 = '1'
                    if(chaos_list[time+2]>= Threshold_2[user]['TH3_01']):
                        j3 = '1'
                        if(chaos_list[time+3]>= Threshold_2[user]['TH4_011']):
                            j4 = '1'
                            j = j1+j2+j3+j4
                            earn = bernoulli.rvs(signal_path_2[user][j][0])
                            signal_path_2[user][j][1] = signal_path_2[user][j][1] + 1 
                            signal_path_2[user][j][2] = signal_path_2[user][j][2] + earn
                            #priv_2[user]['path'] = j
                            # priv_fuzz[user]['ave_earn'][j] = signal_path_2[user][j][2]/signal_path_2[user][j][1]
                            priv_2[user]['ave_earn'][j] = signal_path_2[user][j][2]/signal_path_2[user][j][1]
                            if(earn > 0):
                                Threshold_2[user]['TH1'] = alpha*Threshold_2[user]['TH1'] + Threshold_2[user]['lam'] 
                                Threshold_2[user]['TH2_0'] = alpha*Threshold_2[user]['TH2_0'] - Threshold_2[user]['lam']
                                Threshold_2[user]['TH3_01'] = alpha*Threshold_2[user]['TH3_01'] - Threshold_2[user]['lam']
                                Threshold_2[user]['TH4_011'] = alpha*Threshold_2[user]['TH4_011'] - Threshold_2[user]['lam']
                            else:
                                Threshold_2[user]['TH1'] = alpha*Threshold_2[user]['TH1'] - Threshold_2[user]['ome']
                                Threshold_2[user]['TH2_0'] = alpha*Threshold_2[user]['TH2_0'] + Threshold_2[user]['ome']
                                Threshold_2[user]['TH3_01'] = alpha*Threshold_2[user]['TH3_01'] + Threshold_2[user]['ome']
                                Threshold_2[user]['TH4_011'] = alpha*Threshold_2[user]['TH4_011'] + Threshold_2[user]['ome'] 
                            Threshold_2[user]['TH1'] = np.clip(Threshold_2[user]['TH1'],-Z,Z)
                            Threshold_2[user]['TH2_0'] = np.clip(Threshold_2[user]['TH2_0'],-Z,Z)
                            Threshold_2[user]['TH3_01'] = np.clip(Threshold_2[user]['TH3_01'],-Z,Z)
                            Threshold_2[user]['TH4_011'] = np.clip(Threshold_2[user]['TH4_011'],-Z,Z)
                        else:
                            j4 = '0'
                            j = j1+j2+j3+j4
                            earn = bernoulli.rvs(signal_path_2[user][j][0])
                            signal_path_2[user][j][1] = signal_path_2[user][j][1] + 1 
                            signal_path_2[user][j][2] = signal_path_2[user][j][2] + earn
                            #priv_2[user]['path'] = j
                            # priv_fuzz[user]['ave_earn'][j] = signal_path_2[user][j][2]/signal_path_2[user][j][1]
                            priv_2[user]['ave_earn'][j] = signal_path_2[user][j][2]/signal_path_2[user][j][1]
                            if(earn > 0):
                                Threshold_2[user]['TH1'] = alpha*Threshold_2[user]['TH1'] + Threshold_2[user]['lam'] 
                                Threshold_2[user]['TH2_0'] = alpha*Threshold_2[user]['TH2_0'] - Threshold_2[user]['lam']
                                Threshold_2[user]['TH3_01'] = alpha*Threshold_2[user]['TH3_01'] - Threshold_2[user]['lam']
                                Threshold_2[user]['TH4_011'] = alpha*Threshold_2[user]['TH4_011'] + Threshold_2[user]['lam']
                            else:
                                Threshold_2[user]['TH1'] = alpha*Threshold_2[user]['TH1'] - Threshold_2[user]['ome']
                                Threshold_2[user]['TH2_0'] = alpha*Threshold_2[user]['TH2_0'] + Threshold_2[user]['ome']
                                Threshold_2[user]['TH3_01'] = alpha*Threshold_2[user]['TH3_01'] + Threshold_2[user]['ome']
                                Threshold_2[user]['TH4_011'] = alpha*Threshold_2[user]['TH4_011'] - Threshold_2[user]['ome'] 
                            Threshold_2[user]['TH1'] = np.clip(Threshold_2[user]['TH1'],-Z,Z)
                            Threshold_2[user]['TH2_0'] = np.clip(Threshold_2[user]['TH2_0'],-Z,Z)
                            Threshold_2[user]['TH3_01'] = np.clip(Threshold_2[user]['TH3_01'],-Z,Z)
                            Threshold_2[user]['TH4_011'] = np.clip(Threshold_2[user]['TH4_011'],-Z,Z)
                    else:
                        j3 = '0'
                        if(chaos_list[time+3]>= Threshold_2[user]['TH4_010']):
                            j4 = '1'
                            j = j1+j2+j3+j4
                            earn = bernoulli.rvs(signal_path_2[user][j][0])
                            signal_path_2[user][j][1] = signal_path_2[user][j][1] + 1 
                            signal_path_2[user][j][2] = signal_path_2[user][j][2] + earn
                            #priv_2[user]['path'] = j
                            # priv_fuzz[user]['ave_earn'][j] = signal_path_2[user][j][2]/signal_path_2[user][j][1]
                            priv_2[user]['ave_earn'][j] = signal_path_2[user][j][2]/signal_path_2[user][j][1]
                            if(earn > 0):
                                Threshold_2[user]['TH1'] = alpha*Threshold_2[user]['TH1'] + Threshold_2[user]['lam'] 
                                Threshold_2[user]['TH2_0'] = alpha*Threshold_2[user]['TH2_0'] - Threshold_2[user]['lam']
                                Threshold_2[user]['TH3_01'] = alpha*Threshold_2[user]['TH3_01'] + Threshold_2[user]['lam']
                                Threshold_2[user]['TH4_010'] = alpha*Threshold_2[user]['TH4_010'] - Threshold_2[user]['lam']
                            else:
                                Threshold_2[user]['TH1'] = alpha*Threshold_2[user]['TH1'] - Threshold_2[user]['ome']
                                Threshold_2[user]['TH2_0'] = alpha*Threshold_2[user]['TH2_0'] + Threshold_2[user]['ome']
                                Threshold_2[user]['TH3_01'] = alpha*Threshold_2[user]['TH3_01'] - Threshold_2[user]['ome']
                                Threshold_2[user]['TH4_010'] = alpha*Threshold_2[user]['TH4_010'] + Threshold_2[user]['ome']
                            Threshold_2[user]['TH1'] = np.clip(Threshold_2[user]['TH1'],-Z,Z)
                            Threshold_2[user]['TH2_0'] = np.clip(Threshold_2[user]['TH2_0'],-Z,Z)
                            Threshold_2[user]['TH3_01'] = np.clip(Threshold_2[user]['TH3_01'],-Z,Z)
                            Threshold_2[user]['TH4_010'] = np.clip(Threshold_2[user]['TH4_010'],-Z,Z)
                        else:
                            j4 = '0'
                            j = j1+j2+j3+j4
                            earn = bernoulli.rvs(signal_path_2[user][j][0])
                            signal_path_2[user][j][1] = signal_path_2[user][j][1] + 1 
                            signal_path_2[user][j][2] = signal_path_2[user][j][2] + earn
                            #priv_2[user]['path'] = j
                            # priv_fuzz[user]['ave_earn'][j] = signal_path_2[user][j][2]/signal_path_2[user][j][1]
                            priv_2[user]['ave_earn'][j] = signal_path_2[user][j][2]/signal_path_2[user][j][1]
                            if(earn > 0):
                                Threshold_2[user]['TH1'] = alpha*Threshold_2[user]['TH1'] + Threshold_2[user]['lam'] 
                                Threshold_2[user]['TH2_0'] = alpha*Threshold_2[user]['TH2_0'] - Threshold_2[user]['lam']
                                Threshold_2[user]['TH3_01'] = alpha*Threshold_2[user]['TH3_01'] + Threshold_2[user]['lam']
                                Threshold_2[user]['TH4_010'] = alpha*Threshold_2[user]['TH4_010'] + Threshold_2[user]['lam']
                            else:
                                Threshold_2[user]['TH1'] = alpha*Threshold_2[user]['TH1'] - Threshold_2[user]['ome']
                                Threshold_2[user]['TH2_0'] = alpha*Threshold_2[user]['TH2_0'] + Threshold_2[user]['ome']
                                Threshold_2[user]['TH3_01'] = alpha*Threshold_2[user]['TH3_01'] - Threshold_2[user]['ome']
                                Threshold_2[user]['TH4_010'] = alpha*Threshold_2[user]['TH4_010'] - Threshold_2[user]['ome']
                            Threshold_2[user]['TH1'] = np.clip(Threshold_2[user]['TH1'],-Z,Z)
                            Threshold_2[user]['TH2_0'] = np.clip(Threshold_2[user]['TH2_0'],-Z,Z)
                            Threshold_2[user]['TH3_01'] = np.clip(Threshold_2[user]['TH3_01'],-Z,Z)
                            Threshold_2[user]['TH4_010'] = np.clip(Threshold_2[user]['TH4_010'],-Z,Z)
                else:
                    j2 = '0'
                    if(chaos_list[time+2]>= Threshold_2[user]['TH3_00']):
                        j3 = '1'
                        if(chaos_list[time+3]>= Threshold_2[user]['TH4_001']):
                            j4 = '1'
                            j = j1+j2+j3+j4
                            earn = bernoulli.rvs(signal_path_2[user][j][0])
                            signal_path_2[user][j][1] = signal_path_2[user][j][1] + 1 
                            signal_path_2[user][j][2] = signal_path_2[user][j][2] + earn
                            #priv_2[user]['path'] = j
                            # priv_fuzz[user]['ave_earn'][j] = signal_path_2[user][j][2]/signal_path_2[user][j][1]
                            priv_2[user]['ave_earn'][j] = signal_path_2[user][j][2]/signal_path_2[user][j][1]
                            if(earn > 0):
                                Threshold_2[user]['TH1'] = alpha*Threshold_2[user]['TH1'] + Threshold_2[user]['lam'] 
                                Threshold_2[user]['TH2_0'] = alpha*Threshold_2[user]['TH2_0'] + Threshold_2[user]['lam']
                                Threshold_2[user]['TH3_00'] = alpha*Threshold_2[user]['TH3_00'] - Threshold_2[user]['lam']
                                Threshold_2[user]['TH4_001'] = alpha*Threshold_2[user]['TH4_001'] - Threshold_2[user]['lam']
                            else:
                                Threshold_2[user]['TH1'] = alpha*Threshold_2[user]['TH1'] - Threshold_2[user]['ome']
                                Threshold_2[user]['TH2_0'] = alpha*Threshold_2[user]['TH2_0'] - Threshold_2[user]['ome']
                                Threshold_2[user]['TH3_00'] = alpha*Threshold_2[user]['TH3_00'] + Threshold_2[user]['ome']
                                Threshold_2[user]['TH4_001'] = alpha*Threshold_2[user]['TH4_001'] + Threshold_2[user]['ome']
                            Threshold_2[user]['TH1'] = np.clip(Threshold_2[user]['TH1'],-Z,Z)
                            Threshold_2[user]['TH2_0'] = np.clip(Threshold_2[user]['TH2_0'],-Z,Z)
                            Threshold_2[user]['TH3_00'] = np.clip(Threshold_2[user]['TH3_00'],-Z,Z)
                            Threshold_2[user]['TH4_001'] = np.clip(Threshold_2[user]['TH4_001'],-Z,Z)
                        else:
                            j4 = '0'
                            j = j1+j2+j3+j4
                            earn = bernoulli.rvs(signal_path_2[user][j][0])
                            signal_path_2[user][j][1] = signal_path_2[user][j][1] + 1 
                            signal_path_2[user][j][2] = signal_path_2[user][j][2] + earn
                            #priv_2[user]['path'] = j
                            # priv_fuzz[user]['ave_earn'][j] = signal_path_2[user][j][2]/signal_path_2[user][j][1]
                            priv_2[user]['ave_earn'][j] = signal_path_2[user][j][2]/signal_path_2[user][j][1]
                            if(earn > 0):
                                Threshold_2[user]['TH1'] = alpha*Threshold_2[user]['TH1'] + Threshold_2[user]['lam'] 
                                Threshold_2[user]['TH2_0'] = alpha*Threshold_2[user]['TH2_0'] + Threshold_2[user]['lam']
                                Threshold_2[user]['TH3_00'] = alpha*Threshold_2[user]['TH3_00'] - Threshold_2[user]['lam']
                                Threshold_2[user]['TH4_001'] = alpha*Threshold_2[user]['TH4_001'] + Threshold_2[user]['lam']
                            else:
                                Threshold_2[user]['TH1'] = alpha*Threshold_2[user]['TH1'] - Threshold_2[user]['ome']
                                Threshold_2[user]['TH2_0'] = alpha*Threshold_2[user]['TH2_0'] - Threshold_2[user]['ome']
                                Threshold_2[user]['TH3_00'] = alpha*Threshold_2[user]['TH3_00'] + Threshold_2[user]['ome']
                                Threshold_2[user]['TH4_001'] = alpha*Threshold_2[user]['TH4_001'] - Threshold_2[user]['ome']
                            Threshold_2[user]['TH1'] = np.clip(Threshold_2[user]['TH1'],-Z,Z)
                            Threshold_2[user]['TH2_0'] = np.clip(Threshold_2[user]['TH2_0'],-Z,Z)
                            Threshold_2[user]['TH3_00'] = np.clip(Threshold_2[user]['TH3_00'],-Z,Z)
                            Threshold_2[user]['TH4_001'] = np.clip(Threshold_2[user]['TH4_001'],-Z,Z)
                    else:
                        j3 = '0'
                        if(chaos_list[time+3]>= Threshold_2[user]['TH4_000']):
                            j4 = '1'
                            j = j1+j2+j3+j4
                            earn = bernoulli.rvs(signal_path_2[user][j][0])
                            signal_path_2[user][j][1] = signal_path_2[user][j][1] + 1 
                            signal_path_2[user][j][2] = signal_path_2[user][j][2] + earn
                            #priv_2[user]['path'] = j
                            # priv_fuzz[user]['ave_earn'][j] = signal_path_2[user][j][2]/signal_path_2[user][j][1]
                            priv_2[user]['ave_earn'][j] = signal_path_2[user][j][2]/signal_path_2[user][j][1]
                            if(earn > 0):
                                Threshold_2[user]['TH1'] = alpha*Threshold_2[user]['TH1'] + Threshold_2[user]['lam'] 
                                Threshold_2[user]['TH2_0'] = alpha*Threshold_2[user]['TH2_0'] + Threshold_2[user]['lam']
                                Threshold_2[user]['TH3_00'] = alpha*Threshold_2[user]['TH3_00'] + Threshold_2[user]['lam']
                                Threshold_2[user]['TH4_000'] = alpha*Threshold_2[user]['TH4_000'] - Threshold_2[user]['lam']
                            else:
                                Threshold_2[user]['TH1'] = alpha*Threshold_2[user]['TH1'] - Threshold_2[user]['ome']
                                Threshold_2[user]['TH2_0'] = alpha*Threshold_2[user]['TH2_0'] - Threshold_2[user]['ome']
                                Threshold_2[user]['TH3_00'] = alpha*Threshold_2[user]['TH3_00'] - Threshold_2[user]['ome']
                                Threshold_2[user]['TH4_000'] = alpha*Threshold_2[user]['TH4_000'] + Threshold_2[user]['ome']
                            Threshold_2[user]['TH1'] = np.clip(Threshold_2[user]['TH1'],-Z,Z)
                            Threshold_2[user]['TH2_0'] = np.clip(Threshold_2[user]['TH2_0'],-Z,Z)
                            Threshold_2[user]['TH3_00'] = np.clip(Threshold_2[user]['TH3_00'],-Z,Z)
                            Threshold_2[user]['TH4_000'] = np.clip(Threshold_2[user]['TH4_000'],-Z,Z)
                        else:
                            j4 = '0'
                            j = j1+j2+j3+j4
                            earn = bernoulli.rvs(signal_path_2[user][j][0])
                            signal_path_2[user][j][1] = signal_path_2[user][j][1] + 1 
                            signal_path_2[user][j][2] = signal_path_2[user][j][2] + earn
                            #priv_2[user]['path'] = j
                            # priv_fuzz[user]['ave_earn'][j] = signal_path_2[user][j][2]/signal_path_2[user][j][1]
                            priv_2[user]['ave_earn'][j] = signal_path_2[user][j][2]/signal_path_2[user][j][1]
                            if(earn > 0):
                                Threshold_2[user]['TH1'] = alpha*Threshold_2[user]['TH1'] + Threshold_2[user]['lam'] 
                                Threshold_2[user]['TH2_0'] = alpha*Threshold_2[user]['TH2_0'] + Threshold_2[user]['lam']
                                Threshold_2[user]['TH3_00'] = alpha*Threshold_2[user]['TH3_00'] + Threshold_2[user]['lam']
                                Threshold_2[user]['TH4_000'] = alpha*Threshold_2[user]['TH4_000'] + Threshold_2[user]['lam']
                            else:
                                Threshold_2[user]['TH1'] = alpha*Threshold_2[user]['TH1'] - Threshold_2[user]['ome']
                                Threshold_2[user]['TH2_0'] = alpha*Threshold_2[user]['TH2_0'] - Threshold_2[user]['ome']
                                Threshold_2[user]['TH3_00'] = alpha*Threshold_2[user]['TH3_00'] - Threshold_2[user]['ome']
                                Threshold_2[user]['TH4_000'] = alpha*Threshold_2[user]['TH4_000'] - Threshold_2[user]['ome']
                            Threshold_2[user]['TH1'] = np.clip(Threshold_2[user]['TH1'],-Z,Z)
                            Threshold_2[user]['TH2_0'] = np.clip(Threshold_2[user]['TH2_0'],-Z,Z)
                            Threshold_2[user]['TH3_00'] = np.clip(Threshold_2[user]['TH3_00'],-Z,Z)
                            Threshold_2[user]['TH4_000'] = np.clip(Threshold_2[user]['TH4_000'],-Z,Z)
            seq_2 = []
            prefs = sorted(priv_2[user]['ave_earn'].items(),key = lambda x:x[1],reverse= True)
            for pref in prefs:
                seq_2.append(pref[0])
            priv_2[user]['pref'] = seq_2

            chaos1 = np.random.normal(0,1)
            chaos2 = np.random.normal(0,1)
            chaos3 = np.random.normal(0,1)
            chaos4 = np.random.normal(0,1)

            if(chao1>= Threshold_3[user]['TH1']):
                j1 = '1'       
                if(chao2>= Threshold_3[user]['TH2_1']):
                    j2 = '1'
                    if(chao3>= Threshold_3[user]['TH3_11']):
                        j3 = '1'
                        if(chao4>= Threshold_3[user]['TH4_111']):
                            j4 = '1'
                            j = j1+j2+j3+j4
                            earn = bernoulli.rvs(signal_path_3[user][j][0])
                            signal_path_3[user][j][1] = signal_path_3[user][j][1] + 1 
                            signal_path_3[user][j][2] = signal_path_3[user][j][2] + earn
                            #priv_1[user]['path'] = j
                            priv_3[user]['ave_earn'][j] = signal_path_3[user][j][2]/signal_path_3[user][j][1]
                            # priv_fuzz[user]['ave_earn'][j] = signal_path_3[user][j][2]/signal_path_3[user][j][1]
                            if(earn > 0):
                                Threshold_3[user]['TH1'] = alpha*Threshold_3[user]['TH1'] - Threshold_3[user]['lam'] 
                                Threshold_3[user]['TH2_1'] = alpha*Threshold_3[user]['TH2_1'] - Threshold_3[user]['lam']
                                Threshold_3[user]['TH3_11'] = alpha*Threshold_3[user]['TH3_11'] - Threshold_3[user]['lam']
                                Threshold_3[user]['TH4_111'] = alpha*Threshold_3[user]['TH4_111'] - Threshold_3[user]['lam']
                            else:
                                Threshold_3[user]['TH1'] = alpha*Threshold_3[user]['TH1'] + Threshold_3[user]['ome']
                                Threshold_3[user]['TH2_1'] = alpha*Threshold_3[user]['TH2_1'] + Threshold_3[user]['ome']
                                Threshold_3[user]['TH3_11'] = alpha*Threshold_3[user]['TH3_11'] + Threshold_3[user]['ome']
                                Threshold_3[user]['TH4_111'] = alpha*Threshold_3[user]['TH4_111'] + Threshold_3[user]['ome'] 
                            Threshold_3[user]['TH1'] = np.clip(Threshold_3[user]['TH1'],-Z,Z)
                            Threshold_3[user]['TH2_1'] = np.clip(Threshold_3[user]['TH2_1'],-Z,Z)
                            Threshold_3[user]['TH3_11'] = np.clip(Threshold_3[user]['TH3_11'],-Z,Z)
                            Threshold_3[user]['TH4_111'] = np.clip(Threshold_3[user]['TH4_111'],-Z,Z)
                        else:
                            j4 = '0'
                            j = j1+j2+j3+j4
                            earn = bernoulli.rvs(signal_path_3[user][j][0])
                            signal_path_3[user][j][1] = signal_path_3[user][j][1] + 1 
                            signal_path_3[user][j][2] = signal_path_3[user][j][2] + earn
                            #priv_3[user]['path'] = j
                            # priv_fuzz[user]['ave_earn'][j] = signal_path_3[user][j][2]/signal_path_3[user][j][1]
                            priv_3[user]['ave_earn'][j] = signal_path_3[user][j][2]/signal_path_3[user][j][1]
                            if(earn > 0):
                                Threshold_3[user]['TH1'] = alpha*Threshold_3[user]['TH1'] - Threshold_3[user]['lam'] 
                                Threshold_3[user]['TH2_1'] = alpha*Threshold_3[user]['TH2_1'] - Threshold_3[user]['lam']
                                Threshold_3[user]['TH3_11'] = alpha*Threshold_3[user]['TH3_11'] - Threshold_3[user]['lam']
                                Threshold_3[user]['TH4_111'] = alpha*Threshold_3[user]['TH4_111'] + Threshold_3[user]['lam']
                            else:
                                Threshold_3[user]['TH1'] = alpha*Threshold_3[user]['TH1'] + Threshold_3[user]['ome']
                                Threshold_3[user]['TH2_1'] = alpha*Threshold_3[user]['TH2_1'] + Threshold_3[user]['ome']
                                Threshold_3[user]['TH3_11'] = alpha*Threshold_3[user]['TH3_11'] + Threshold_3[user]['ome']
                                Threshold_3[user]['TH4_111'] = alpha*Threshold_3[user]['TH4_111'] - Threshold_3[user]['ome'] 
                            Threshold_3[user]['TH1'] = np.clip(Threshold_3[user]['TH1'],-Z,Z)
                            Threshold_3[user]['TH2_1'] = np.clip(Threshold_3[user]['TH2_1'],-Z,Z)
                            Threshold_3[user]['TH3_11'] = np.clip(Threshold_3[user]['TH3_11'],-Z,Z)
                            Threshold_3[user]['TH4_111'] = np.clip(Threshold_3[user]['TH4_111'],-Z,Z)
                    else:
                        j3 = '0'
                        if(chao4>= Threshold_3[user]['TH4_110']):
                            j4 = '1'
                            j = j1+j2+j3+j4
                            earn = bernoulli.rvs(signal_path_3[user][j][0])
                            signal_path_3[user][j][1] = signal_path_3[user][j][1] + 1 
                            signal_path_3[user][j][2] = signal_path_3[user][j][2] + earn
                            #priv_3[user]['path'] = j
                            # priv_fuzz[user]['ave_earn'][j] = signal_path_3[user][j][2]/signal_path_3[user][j][1]
                            priv_3[user]['ave_earn'][j] = signal_path_3[user][j][2]/signal_path_3[user][j][1]
                            if(earn > 0):
                                Threshold_3[user]['TH1'] = alpha*Threshold_3[user]['TH1'] - Threshold_3[user]['lam'] 
                                Threshold_3[user]['TH2_1'] = alpha*Threshold_3[user]['TH2_1'] - Threshold_3[user]['lam']
                                Threshold_3[user]['TH3_11'] = alpha*Threshold_3[user]['TH3_11'] + Threshold_3[user]['lam']
                                Threshold_3[user]['TH4_110'] = alpha*Threshold_3[user]['TH4_110'] - Threshold_3[user]['lam']
                            else:
                                Threshold_3[user]['TH1'] = alpha*Threshold_3[user]['TH1'] + Threshold_3[user]['ome']
                                Threshold_3[user]['TH2_1'] = alpha*Threshold_3[user]['TH2_1'] + Threshold_3[user]['ome']
                                Threshold_3[user]['TH3_11'] = alpha*Threshold_3[user]['TH3_11'] - Threshold_3[user]['ome']
                                Threshold_3[user]['TH4_110'] = alpha*Threshold_3[user]['TH4_110'] + Threshold_3[user]['ome'] 
                            Threshold_3[user]['TH1'] = np.clip(Threshold_3[user]['TH1'],-Z,Z)
                            Threshold_3[user]['TH2_1'] = np.clip(Threshold_3[user]['TH2_1'],-Z,Z)
                            Threshold_3[user]['TH3_11'] = np.clip(Threshold_3[user]['TH3_11'],-Z,Z)
                            Threshold_3[user]['TH4_110'] = np.clip(Threshold_3[user]['TH4_110'],-Z,Z)
                        else:
                            j4 = '0'
                            j = j1+j2+j3+j4
                            earn = bernoulli.rvs(signal_path_3[user][j][0])
                            signal_path_3[user][j][1] = signal_path_3[user][j][1] + 1 
                            signal_path_3[user][j][2] = signal_path_3[user][j][2] + earn
                            #priv_3[user]['path'] = j
                            # priv_fuzz[user]['ave_earn'][j] = signal_path_3[user][j][2]/signal_path_3[user][j][1]
                            priv_3[user]['ave_earn'][j] = signal_path_3[user][j][2]/signal_path_3[user][j][1]
                            if(earn > 0):
                                Threshold_3[user]['TH1'] = alpha*Threshold_3[user]['TH1'] - Threshold_3[user]['lam'] 
                                Threshold_3[user]['TH2_1'] = alpha*Threshold_3[user]['TH2_1'] - Threshold_3[user]['lam']
                                Threshold_3[user]['TH3_11'] = alpha*Threshold_3[user]['TH3_11'] + Threshold_3[user]['lam']
                                Threshold_3[user]['TH4_110'] = alpha*Threshold_3[user]['TH4_110'] + Threshold_3[user]['lam']
                            else:
                                Threshold_3[user]['TH1'] = alpha*Threshold_3[user]['TH1'] + Threshold_3[user]['ome']
                                Threshold_3[user]['TH2_1'] = alpha*Threshold_3[user]['TH2_1'] + Threshold_3[user]['ome']
                                Threshold_3[user]['TH3_11'] = alpha*Threshold_3[user]['TH3_11'] - Threshold_3[user]['ome']
                                Threshold_3[user]['TH4_110'] = alpha*Threshold_3[user]['TH4_110'] - Threshold_3[user]['ome'] 
                            Threshold_3[user]['TH1'] = np.clip(Threshold_3[user]['TH1'],-Z,Z)
                            Threshold_3[user]['TH2_1'] = np.clip(Threshold_3[user]['TH2_1'],-Z,Z)
                            Threshold_3[user]['TH3_11'] = np.clip(Threshold_3[user]['TH3_11'],-Z,Z)
                            Threshold_3[user]['TH4_110'] = np.clip(Threshold_3[user]['TH4_110'],-Z,Z)                     
                else:
                    j2 = '0'
                    if(chao3>= Threshold_3[user]['TH3_10']):
                        j3 = '1'
                        if(chao4>= Threshold_3[user]['TH4_101']):
                            j4 = '1'
                            j = j1+j2+j3+j4
                            earn = bernoulli.rvs(signal_path_3[user][j][0])
                            signal_path_3[user][j][1] = signal_path_3[user][j][1] + 1 
                            signal_path_3[user][j][2] = signal_path_3[user][j][2] + earn
                            #priv_3[user]['path'] = j
                            # priv_fuzz[user]['ave_earn'][j] = signal_path_3[user][j][2]/signal_path_3[user][j][1]
                            priv_3[user]['ave_earn'][j] = signal_path_3[user][j][2]/signal_path_3[user][j][1]
                            if(earn > 0):
                                Threshold_3[user]['TH1'] = alpha*Threshold_3[user]['TH1'] - Threshold_3[user]['lam'] 
                                Threshold_3[user]['TH2_1'] = alpha*Threshold_3[user]['TH2_1'] + Threshold_3[user]['lam']
                                Threshold_3[user]['TH3_10'] = alpha*Threshold_3[user]['TH3_10'] - Threshold_3[user]['lam']
                                Threshold_3[user]['TH4_101'] = alpha*Threshold_3[user]['TH4_101'] - Threshold_3[user]['lam']
                            else:
                                Threshold_3[user]['TH1'] = alpha*Threshold_3[user]['TH1'] + Threshold_3[user]['ome']
                                Threshold_3[user]['TH2_1'] = alpha*Threshold_3[user]['TH2_1'] - Threshold_3[user]['ome']
                                Threshold_3[user]['TH3_10'] = alpha*Threshold_3[user]['TH3_10'] + Threshold_3[user]['ome']
                                Threshold_3[user]['TH4_101'] = alpha*Threshold_3[user]['TH4_101'] + Threshold_3[user]['ome'] 
                            Threshold_3[user]['TH1'] = np.clip(Threshold_3[user]['TH1'],-Z,Z)
                            Threshold_3[user]['TH2_1'] = np.clip(Threshold_3[user]['TH2_1'],-Z,Z)
                            Threshold_3[user]['TH3_10'] = np.clip(Threshold_3[user]['TH3_10'],-Z,Z)
                            Threshold_3[user]['TH4_101'] = np.clip(Threshold_3[user]['TH4_101'],-Z,Z)
                        else:
                            j4 = '0'
                            j = j1+j2+j3+j4
                            earn = bernoulli.rvs(signal_path_3[user][j][0])
                            signal_path_3[user][j][1] = signal_path_3[user][j][1] + 1 
                            signal_path_3[user][j][2] = signal_path_3[user][j][2] + earn
                            #priv_3[user]['path'] = j
                            # priv_fuzz[user]['ave_earn'][j] = signal_path_3[user][j][2]/signal_path_3[user][j][1]
                            priv_3[user]['ave_earn'][j] = signal_path_3[user][j][2]/signal_path_3[user][j][1]
                            if(earn > 0):
                                Threshold_3[user]['TH1'] = alpha*Threshold_3[user]['TH1'] - Threshold_3[user]['lam'] 
                                Threshold_3[user]['TH2_1'] = alpha*Threshold_3[user]['TH2_1'] + Threshold_3[user]['lam']
                                Threshold_3[user]['TH3_10'] = alpha*Threshold_3[user]['TH3_10'] - Threshold_3[user]['lam']
                                Threshold_3[user]['TH4_101'] = alpha*Threshold_3[user]['TH4_101'] + Threshold_3[user]['lam']
                            else:
                                Threshold_3[user]['TH1'] = alpha*Threshold_3[user]['TH1'] + Threshold_3[user]['ome']
                                Threshold_3[user]['TH2_1'] = alpha*Threshold_3[user]['TH2_1'] - Threshold_3[user]['ome']
                                Threshold_3[user]['TH3_10'] = alpha*Threshold_3[user]['TH3_10'] + Threshold_3[user]['ome']
                                Threshold_3[user]['TH4_101'] = alpha*Threshold_3[user]['TH4_101'] - Threshold_3[user]['ome'] 
                            Threshold_3[user]['TH1'] = np.clip(Threshold_3[user]['TH1'],-Z,Z)
                            Threshold_3[user]['TH2_1'] = np.clip(Threshold_3[user]['TH2_1'],-Z,Z)
                            Threshold_3[user]['TH3_10'] = np.clip(Threshold_3[user]['TH3_10'],-Z,Z)
                            Threshold_3[user]['TH4_101'] = np.clip(Threshold_3[user]['TH4_101'],-Z,Z)
                    else:
                        j3 = '0'
                        if(chao4>= Threshold_3[user]['TH4_100']):
                            j4 = '1'
                            j = j1+j2+j3+j4
                            earn = bernoulli.rvs(signal_path_3[user][j][0])
                            signal_path_3[user][j][1] = signal_path_3[user][j][1] + 1 
                            signal_path_3[user][j][2] = signal_path_3[user][j][2] + earn
                            #priv_3[user]['path'] = j
                            # priv_fuzz[user]['ave_earn'][j] = signal_path_3[user][j][2]/signal_path_3[user][j][1]
                            priv_3[user]['ave_earn'][j] = signal_path_3[user][j][2]/signal_path_3[user][j][1]
                            if(earn > 0):
                                Threshold_3[user]['TH1'] = alpha*Threshold_3[user]['TH1'] - Threshold_3[user]['lam'] 
                                Threshold_3[user]['TH2_1'] = alpha*Threshold_3[user]['TH2_1'] + Threshold_3[user]['lam']
                                Threshold_3[user]['TH3_10'] = alpha*Threshold_3[user]['TH3_10'] + Threshold_3[user]['lam']
                                Threshold_3[user]['TH4_100'] = alpha*Threshold_3[user]['TH4_100'] - Threshold_3[user]['lam']
                            else:
                                Threshold_3[user]['TH1'] = alpha*Threshold_3[user]['TH1'] + Threshold_3[user]['ome']
                                Threshold_3[user]['TH2_1'] = alpha*Threshold_3[user]['TH2_1'] - Threshold_3[user]['ome']
                                Threshold_3[user]['TH3_10'] = alpha*Threshold_3[user]['TH3_10'] - Threshold_3[user]['ome']
                                Threshold_3[user]['TH4_100'] = alpha*Threshold_3[user]['TH4_100'] + Threshold_3[user]['ome'] 
                            Threshold_3[user]['TH1'] = np.clip(Threshold_3[user]['TH1'],-Z,Z)
                            Threshold_3[user]['TH2_1'] = np.clip(Threshold_3[user]['TH2_1'],-Z,Z)
                            Threshold_3[user]['TH3_10'] = np.clip(Threshold_3[user]['TH3_10'],-Z,Z)
                            Threshold_3[user]['TH4_100'] = np.clip(Threshold_3[user]['TH4_100'],-Z,Z)
                        else:
                            j4 = '0'
                            j = j1+j2+j3+j4
                            earn = bernoulli.rvs(signal_path_3[user][j][0])
                            signal_path_3[user][j][1] = signal_path_3[user][j][1] + 1 
                            signal_path_3[user][j][2] = signal_path_3[user][j][2] + earn
                            #priv_3[user]['path'] = j
                            # priv_fuzz[user]['ave_earn'][j] = signal_path_3[user][j][2]/signal_path_3[user][j][1]
                            priv_3[user]['ave_earn'][j] = signal_path_3[user][j][2]/signal_path_3[user][j][1]
                            if(earn > 0):
                                Threshold_3[user]['TH1'] = alpha*Threshold_3[user]['TH1'] - Threshold_3[user]['lam'] 
                                Threshold_3[user]['TH2_1'] = alpha*Threshold_3[user]['TH2_1'] + Threshold_3[user]['lam']
                                Threshold_3[user]['TH3_10'] = alpha*Threshold_3[user]['TH3_10'] + Threshold_3[user]['lam']
                                Threshold_3[user]['TH4_100'] = alpha*Threshold_3[user]['TH4_100'] + Threshold_3[user]['lam']
                            else:
                                Threshold_3[user]['TH1'] = alpha*Threshold_3[user]['TH1'] + Threshold_3[user]['ome']
                                Threshold_3[user]['TH2_1'] = alpha*Threshold_3[user]['TH2_1'] - Threshold_3[user]['ome']
                                Threshold_3[user]['TH3_10'] = alpha*Threshold_3[user]['TH3_10'] - Threshold_3[user]['ome']
                                Threshold_3[user]['TH4_100'] = alpha*Threshold_3[user]['TH4_100'] - Threshold_3[user]['ome'] 
                            Threshold_3[user]['TH1'] = np.clip(Threshold_3[user]['TH1'],-Z,Z)
                            Threshold_3[user]['TH2_1'] = np.clip(Threshold_3[user]['TH2_1'],-Z,Z)
                            Threshold_3[user]['TH3_10'] = np.clip(Threshold_3[user]['TH3_10'],-Z,Z)
                            Threshold_3[user]['TH4_100'] = np.clip(Threshold_3[user]['TH4_100'],-Z,Z)
            else:
                j1 = '0'
                if(chao2>= Threshold_3[user]['TH2_0']):
                    j2 = '1'
                    if(chao3>= Threshold_3[user]['TH3_01']):
                        j3 = '1'
                        if(chao4>= Threshold_3[user]['TH4_011']):
                            j4 = '1'
                            j = j1+j2+j3+j4
                            earn = bernoulli.rvs(signal_path_3[user][j][0])
                            signal_path_3[user][j][1] = signal_path_3[user][j][1] + 1 
                            signal_path_3[user][j][2] = signal_path_3[user][j][2] + earn
                            #priv_3[user]['path'] = j
                            # priv_fuzz[user]['ave_earn'][j] = signal_path_3[user][j][2]/signal_path_3[user][j][1]
                            priv_3[user]['ave_earn'][j] = signal_path_3[user][j][2]/signal_path_3[user][j][1]
                            if(earn > 0):
                                Threshold_3[user]['TH1'] = alpha*Threshold_3[user]['TH1'] + Threshold_3[user]['lam'] 
                                Threshold_3[user]['TH2_0'] = alpha*Threshold_3[user]['TH2_0'] - Threshold_3[user]['lam']
                                Threshold_3[user]['TH3_01'] = alpha*Threshold_3[user]['TH3_01'] - Threshold_3[user]['lam']
                                Threshold_3[user]['TH4_011'] = alpha*Threshold_3[user]['TH4_011'] - Threshold_3[user]['lam']
                            else:
                                Threshold_3[user]['TH1'] = alpha*Threshold_3[user]['TH1'] - Threshold_3[user]['ome']
                                Threshold_3[user]['TH2_0'] = alpha*Threshold_3[user]['TH2_0'] + Threshold_3[user]['ome']
                                Threshold_3[user]['TH3_01'] = alpha*Threshold_3[user]['TH3_01'] + Threshold_3[user]['ome']
                                Threshold_3[user]['TH4_011'] = alpha*Threshold_3[user]['TH4_011'] + Threshold_3[user]['ome'] 
                            Threshold_3[user]['TH1'] = np.clip(Threshold_3[user]['TH1'],-Z,Z)
                            Threshold_3[user]['TH2_0'] = np.clip(Threshold_3[user]['TH2_0'],-Z,Z)
                            Threshold_3[user]['TH3_01'] = np.clip(Threshold_3[user]['TH3_01'],-Z,Z)
                            Threshold_3[user]['TH4_011'] = np.clip(Threshold_3[user]['TH4_011'],-Z,Z)
                        else:
                            j4 = '0'
                            j = j1+j2+j3+j4
                            earn = bernoulli.rvs(signal_path_3[user][j][0])
                            signal_path_3[user][j][1] = signal_path_3[user][j][1] + 1 
                            signal_path_3[user][j][2] = signal_path_3[user][j][2] + earn
                            #priv_3[user]['path'] = j
                            # priv_fuzz[user]['ave_earn'][j] = signal_path_3[user][j][2]/signal_path_3[user][j][1]
                            priv_3[user]['ave_earn'][j] = signal_path_3[user][j][2]/signal_path_3[user][j][1]
                            if(earn > 0):
                                Threshold_3[user]['TH1'] = alpha*Threshold_3[user]['TH1'] + Threshold_3[user]['lam'] 
                                Threshold_3[user]['TH2_0'] = alpha*Threshold_3[user]['TH2_0'] - Threshold_3[user]['lam']
                                Threshold_3[user]['TH3_01'] = alpha*Threshold_3[user]['TH3_01'] - Threshold_3[user]['lam']
                                Threshold_3[user]['TH4_011'] = alpha*Threshold_3[user]['TH4_011'] + Threshold_3[user]['lam']
                            else:
                                Threshold_3[user]['TH1'] = alpha*Threshold_3[user]['TH1'] - Threshold_3[user]['ome']
                                Threshold_3[user]['TH2_0'] = alpha*Threshold_3[user]['TH2_0'] + Threshold_3[user]['ome']
                                Threshold_3[user]['TH3_01'] = alpha*Threshold_3[user]['TH3_01'] + Threshold_3[user]['ome']
                                Threshold_3[user]['TH4_011'] = alpha*Threshold_3[user]['TH4_011'] - Threshold_3[user]['ome'] 
                            Threshold_3[user]['TH1'] = np.clip(Threshold_3[user]['TH1'],-Z,Z)
                            Threshold_3[user]['TH2_0'] = np.clip(Threshold_3[user]['TH2_0'],-Z,Z)
                            Threshold_3[user]['TH3_01'] = np.clip(Threshold_3[user]['TH3_01'],-Z,Z)
                            Threshold_3[user]['TH4_011'] = np.clip(Threshold_3[user]['TH4_011'],-Z,Z)
                    else:
                        j3 = '0'
                        if(chao4>= Threshold_3[user]['TH4_010']):
                            j4 = '1'
                            j = j1+j2+j3+j4
                            earn = bernoulli.rvs(signal_path_3[user][j][0])
                            signal_path_3[user][j][1] = signal_path_3[user][j][1] + 1 
                            signal_path_3[user][j][2] = signal_path_3[user][j][2] + earn
                            #priv_3[user]['path'] = j
                            # priv_fuzz[user]['ave_earn'][j] = signal_path_3[user][j][2]/signal_path_3[user][j][1]
                            priv_3[user]['ave_earn'][j] = signal_path_3[user][j][2]/signal_path_3[user][j][1]
                            if(earn > 0):
                                Threshold_3[user]['TH1'] = alpha*Threshold_3[user]['TH1'] + Threshold_3[user]['lam'] 
                                Threshold_3[user]['TH2_0'] = alpha*Threshold_3[user]['TH2_0'] - Threshold_3[user]['lam']
                                Threshold_3[user]['TH3_01'] = alpha*Threshold_3[user]['TH3_01'] + Threshold_3[user]['lam']
                                Threshold_3[user]['TH4_010'] = alpha*Threshold_3[user]['TH4_010'] - Threshold_3[user]['lam']
                            else:
                                Threshold_3[user]['TH1'] = alpha*Threshold_3[user]['TH1'] - Threshold_3[user]['ome']
                                Threshold_3[user]['TH2_0'] = alpha*Threshold_3[user]['TH2_0'] + Threshold_3[user]['ome']
                                Threshold_3[user]['TH3_01'] = alpha*Threshold_3[user]['TH3_01'] - Threshold_3[user]['ome']
                                Threshold_3[user]['TH4_010'] = alpha*Threshold_3[user]['TH4_010'] + Threshold_3[user]['ome']
                            Threshold_3[user]['TH1'] = np.clip(Threshold_3[user]['TH1'],-Z,Z)
                            Threshold_3[user]['TH2_0'] = np.clip(Threshold_3[user]['TH2_0'],-Z,Z)
                            Threshold_3[user]['TH3_01'] = np.clip(Threshold_3[user]['TH3_01'],-Z,Z)
                            Threshold_3[user]['TH4_010'] = np.clip(Threshold_3[user]['TH4_010'],-Z,Z)
                        else:
                            j4 = '0'
                            j = j1+j2+j3+j4
                            earn = bernoulli.rvs(signal_path_3[user][j][0])
                            signal_path_3[user][j][1] = signal_path_3[user][j][1] + 1 
                            signal_path_3[user][j][2] = signal_path_3[user][j][2] + earn
                            #priv_3[user]['path'] = j
                            # priv_fuzz[user]['ave_earn'][j] = signal_path_3[user][j][2]/signal_path_3[user][j][1]
                            priv_3[user]['ave_earn'][j] = signal_path_3[user][j][2]/signal_path_3[user][j][1]
                            if(earn > 0):
                                Threshold_3[user]['TH1'] = alpha*Threshold_3[user]['TH1'] + Threshold_3[user]['lam'] 
                                Threshold_3[user]['TH2_0'] = alpha*Threshold_3[user]['TH2_0'] - Threshold_3[user]['lam']
                                Threshold_3[user]['TH3_01'] = alpha*Threshold_3[user]['TH3_01'] + Threshold_3[user]['lam']
                                Threshold_3[user]['TH4_010'] = alpha*Threshold_3[user]['TH4_010'] + Threshold_3[user]['lam']
                            else:
                                Threshold_3[user]['TH1'] = alpha*Threshold_3[user]['TH1'] - Threshold_3[user]['ome']
                                Threshold_3[user]['TH2_0'] = alpha*Threshold_3[user]['TH2_0'] + Threshold_3[user]['ome']
                                Threshold_3[user]['TH3_01'] = alpha*Threshold_3[user]['TH3_01'] - Threshold_3[user]['ome']
                                Threshold_3[user]['TH4_010'] = alpha*Threshold_3[user]['TH4_010'] - Threshold_3[user]['ome']
                            Threshold_3[user]['TH1'] = np.clip(Threshold_3[user]['TH1'],-Z,Z)
                            Threshold_3[user]['TH2_0'] = np.clip(Threshold_3[user]['TH2_0'],-Z,Z)
                            Threshold_3[user]['TH3_01'] = np.clip(Threshold_3[user]['TH3_01'],-Z,Z)
                            Threshold_3[user]['TH4_010'] = np.clip(Threshold_3[user]['TH4_010'],-Z,Z)
                else:
                    j2 = '0'
                    if(chao3>= Threshold_3[user]['TH3_00']):
                        j3 = '1'
                        if(chao4>= Threshold_3[user]['TH4_001']):
                            j4 = '1'
                            j = j1+j2+j3+j4
                            earn = bernoulli.rvs(signal_path_3[user][j][0])
                            signal_path_3[user][j][1] = signal_path_3[user][j][1] + 1 
                            signal_path_3[user][j][2] = signal_path_3[user][j][2] + earn
                            #priv_3[user]['path'] = j
                            # priv_fuzz[user]['ave_earn'][j] = signal_path_3[user][j][2]/signal_path_3[user][j][1]
                            priv_3[user]['ave_earn'][j] = signal_path_3[user][j][2]/signal_path_3[user][j][1]
                            if(earn > 0):
                                Threshold_3[user]['TH1'] = alpha*Threshold_3[user]['TH1'] + Threshold_3[user]['lam'] 
                                Threshold_3[user]['TH2_0'] = alpha*Threshold_3[user]['TH2_0'] + Threshold_3[user]['lam']
                                Threshold_3[user]['TH3_00'] = alpha*Threshold_3[user]['TH3_00'] - Threshold_3[user]['lam']
                                Threshold_3[user]['TH4_001'] = alpha*Threshold_3[user]['TH4_001'] - Threshold_3[user]['lam']
                            else:
                                Threshold_3[user]['TH1'] = alpha*Threshold_3[user]['TH1'] - Threshold_3[user]['ome']
                                Threshold_3[user]['TH2_0'] = alpha*Threshold_3[user]['TH2_0'] - Threshold_3[user]['ome']
                                Threshold_3[user]['TH3_00'] = alpha*Threshold_3[user]['TH3_00'] + Threshold_3[user]['ome']
                                Threshold_3[user]['TH4_001'] = alpha*Threshold_3[user]['TH4_001'] + Threshold_3[user]['ome']
                            Threshold_3[user]['TH1'] = np.clip(Threshold_3[user]['TH1'],-Z,Z)
                            Threshold_3[user]['TH2_0'] = np.clip(Threshold_3[user]['TH2_0'],-Z,Z)
                            Threshold_3[user]['TH3_00'] = np.clip(Threshold_3[user]['TH3_00'],-Z,Z)
                            Threshold_3[user]['TH4_001'] = np.clip(Threshold_3[user]['TH4_001'],-Z,Z)
                        else:
                            j4 = '0'
                            j = j1+j2+j3+j4
                            earn = bernoulli.rvs(signal_path_3[user][j][0])
                            signal_path_3[user][j][1] = signal_path_3[user][j][1] + 1 
                            signal_path_3[user][j][2] = signal_path_3[user][j][2] + earn
                            #priv_3[user]['path'] = j
                            # priv_fuzz[user]['ave_earn'][j] = signal_path_3[user][j][2]/signal_path_3[user][j][1]
                            priv_3[user]['ave_earn'][j] = signal_path_3[user][j][2]/signal_path_3[user][j][1]
                            if(earn > 0):
                                Threshold_3[user]['TH1'] = alpha*Threshold_3[user]['TH1'] + Threshold_3[user]['lam'] 
                                Threshold_3[user]['TH2_0'] = alpha*Threshold_3[user]['TH2_0'] + Threshold_3[user]['lam']
                                Threshold_3[user]['TH3_00'] = alpha*Threshold_3[user]['TH3_00'] - Threshold_3[user]['lam']
                                Threshold_3[user]['TH4_001'] = alpha*Threshold_3[user]['TH4_001'] + Threshold_3[user]['lam']
                            else:
                                Threshold_3[user]['TH1'] = alpha*Threshold_3[user]['TH1'] - Threshold_3[user]['ome']
                                Threshold_3[user]['TH2_0'] = alpha*Threshold_3[user]['TH2_0'] - Threshold_3[user]['ome']
                                Threshold_3[user]['TH3_00'] = alpha*Threshold_3[user]['TH3_00'] + Threshold_3[user]['ome']
                                Threshold_3[user]['TH4_001'] = alpha*Threshold_3[user]['TH4_001'] - Threshold_3[user]['ome']
                            Threshold_3[user]['TH1'] = np.clip(Threshold_3[user]['TH1'],-Z,Z)
                            Threshold_3[user]['TH2_0'] = np.clip(Threshold_3[user]['TH2_0'],-Z,Z)
                            Threshold_3[user]['TH3_00'] = np.clip(Threshold_3[user]['TH3_00'],-Z,Z)
                            Threshold_3[user]['TH4_001'] = np.clip(Threshold_3[user]['TH4_001'],-Z,Z)
                    else:
                        j3 = '0'
                        if(chao4>= Threshold_3[user]['TH4_000']):
                            j4 = '1'
                            j = j1+j2+j3+j4
                            earn = bernoulli.rvs(signal_path_3[user][j][0])
                            signal_path_3[user][j][1] = signal_path_3[user][j][1] + 1 
                            signal_path_3[user][j][2] = signal_path_3[user][j][2] + earn
                            #priv_3[user]['path'] = j
                            # priv_fuzz[user]['ave_earn'][j] = signal_path_3[user][j][2]/signal_path_3[user][j][1]
                            priv_3[user]['ave_earn'][j] = signal_path_3[user][j][2]/signal_path_3[user][j][1]
                            if(earn > 0):
                                Threshold_3[user]['TH1'] = alpha*Threshold_3[user]['TH1'] + Threshold_3[user]['lam'] 
                                Threshold_3[user]['TH2_0'] = alpha*Threshold_3[user]['TH2_0'] + Threshold_3[user]['lam']
                                Threshold_3[user]['TH3_00'] = alpha*Threshold_3[user]['TH3_00'] + Threshold_3[user]['lam']
                                Threshold_3[user]['TH4_000'] = alpha*Threshold_3[user]['TH4_000'] - Threshold_3[user]['lam']
                            else:
                                Threshold_3[user]['TH1'] = alpha*Threshold_3[user]['TH1'] - Threshold_3[user]['ome']
                                Threshold_3[user]['TH2_0'] = alpha*Threshold_3[user]['TH2_0'] - Threshold_3[user]['ome']
                                Threshold_3[user]['TH3_00'] = alpha*Threshold_3[user]['TH3_00'] - Threshold_3[user]['ome']
                                Threshold_3[user]['TH4_000'] = alpha*Threshold_3[user]['TH4_000'] + Threshold_3[user]['ome']
                            Threshold_3[user]['TH1'] = np.clip(Threshold_3[user]['TH1'],-Z,Z)
                            Threshold_3[user]['TH2_0'] = np.clip(Threshold_3[user]['TH2_0'],-Z,Z)
                            Threshold_3[user]['TH3_00'] = np.clip(Threshold_3[user]['TH3_00'],-Z,Z)
                            Threshold_3[user]['TH4_000'] = np.clip(Threshold_3[user]['TH4_000'],-Z,Z)
                        else:
                            j4 = '0'
                            j = j1+j2+j3+j4
                            earn = bernoulli.rvs(signal_path_3[user][j][0])
                            signal_path_3[user][j][1] = signal_path_3[user][j][1] + 1 
                            signal_path_3[user][j][2] = signal_path_3[user][j][2] + earn
                            #priv_3[user]['path'] = j
                            # priv_fuzz[user]['ave_earn'][j] = signal_path_3[user][j][2]/signal_path_3[user][j][1]
                            priv_3[user]['ave_earn'][j] = signal_path_3[user][j][2]/signal_path_3[user][j][1]
                            if(earn > 0):
                                Threshold_3[user]['TH1'] = alpha*Threshold_3[user]['TH1'] + Threshold_3[user]['lam'] 
                                Threshold_3[user]['TH2_0'] = alpha*Threshold_3[user]['TH2_0'] + Threshold_3[user]['lam']
                                Threshold_3[user]['TH3_00'] = alpha*Threshold_3[user]['TH3_00'] + Threshold_3[user]['lam']
                                Threshold_3[user]['TH4_000'] = alpha*Threshold_3[user]['TH4_000'] + Threshold_3[user]['lam']
                            else:
                                Threshold_3[user]['TH1'] = alpha*Threshold_3[user]['TH1'] - Threshold_3[user]['ome']
                                Threshold_3[user]['TH2_0'] = alpha*Threshold_3[user]['TH2_0'] - Threshold_3[user]['ome']
                                Threshold_3[user]['TH3_00'] = alpha*Threshold_3[user]['TH3_00'] - Threshold_3[user]['ome']
                                Threshold_3[user]['TH4_000'] = alpha*Threshold_3[user]['TH4_000'] - Threshold_3[user]['ome']
                            Threshold_3[user]['TH1'] = np.clip(Threshold_3[user]['TH1'],-Z,Z)
                            Threshold_3[user]['TH2_0'] = np.clip(Threshold_3[user]['TH2_0'],-Z,Z)
                            Threshold_3[user]['TH3_00'] = np.clip(Threshold_3[user]['TH3_00'],-Z,Z)
                            Threshold_3[user]['TH4_000'] = np.clip(Threshold_3[user]['TH4_000'],-Z,Z)
            seq_3 = []
            prefs = sorted(priv_3[user]['ave_earn'].items(),key = lambda x:x[1],reverse= True)
            for pref in prefs:
                seq_3.append(pref[0])
            priv_3[user]['pref'] = seq_3

            ##第四种随机数
            rand1 = Z*(2*random.random()-1)
            rand2 = Z*(2*random.random()-1)
            rand3 = Z*(2*random.random()-1)
            rand4 = Z*(2*random.random()-1)
            

            if(rand1>= Threshold_4[user]['TH1']):
                j1 = '1'       
                if(rand2>= Threshold_4[user]['TH2_1']):
                    j2 = '1'
                    if(rand3>= Threshold_4[user]['TH3_11']):
                        j3 = '1'
                        if(rand4>= Threshold_4[user]['TH4_111']):
                            j4 = '1'
                            j = j1+j2+j3+j4
                            earn = bernoulli.rvs(signal_path_4[user][j][0])
                            signal_path_4[user][j][1] = signal_path_4[user][j][1] + 1 
                            signal_path_4[user][j][2] = signal_path_4[user][j][2] + earn
                            #priv_1[user]['path'] = j
                            priv_4[user]['ave_earn'][j] = signal_path_4[user][j][2]/signal_path_4[user][j][1]
                            # priv_fuzz[user]['ave_earn'][j] = signal_path_4[user][j][2]/signal_path_4[user][j][1]
                            if(earn > 0):
                                Threshold_4[user]['TH1'] = alpha*Threshold_4[user]['TH1'] - Threshold_4[user]['lam'] 
                                Threshold_4[user]['TH2_1'] = alpha*Threshold_4[user]['TH2_1'] - Threshold_4[user]['lam']
                                Threshold_4[user]['TH3_11'] = alpha*Threshold_4[user]['TH3_11'] - Threshold_4[user]['lam']
                                Threshold_4[user]['TH4_111'] = alpha*Threshold_4[user]['TH4_111'] - Threshold_4[user]['lam']
                            else:
                                Threshold_4[user]['TH1'] = alpha*Threshold_4[user]['TH1'] + Threshold_4[user]['ome']
                                Threshold_4[user]['TH2_1'] = alpha*Threshold_4[user]['TH2_1'] + Threshold_4[user]['ome']
                                Threshold_4[user]['TH3_11'] = alpha*Threshold_4[user]['TH3_11'] + Threshold_4[user]['ome']
                                Threshold_4[user]['TH4_111'] = alpha*Threshold_4[user]['TH4_111'] + Threshold_4[user]['ome'] 
                            Threshold_4[user]['TH1'] = np.clip(Threshold_4[user]['TH1'],-Z,Z)
                            Threshold_4[user]['TH2_1'] = np.clip(Threshold_4[user]['TH2_1'],-Z,Z)
                            Threshold_4[user]['TH3_11'] = np.clip(Threshold_4[user]['TH3_11'],-Z,Z)
                            Threshold_4[user]['TH4_111'] = np.clip(Threshold_4[user]['TH4_111'],-Z,Z)
                        else:
                            j4 = '0'
                            j = j1+j2+j3+j4
                            earn = bernoulli.rvs(signal_path_4[user][j][0])
                            signal_path_4[user][j][1] = signal_path_4[user][j][1] + 1 
                            signal_path_4[user][j][2] = signal_path_4[user][j][2] + earn
                            #priv_4[user]['path'] = j
                            # priv_fuzz[user]['ave_earn'][j] = signal_path_4[user][j][2]/signal_path_4[user][j][1]
                            priv_4[user]['ave_earn'][j] = signal_path_4[user][j][2]/signal_path_4[user][j][1]
                            if(earn > 0):
                                Threshold_4[user]['TH1'] = alpha*Threshold_4[user]['TH1'] - Threshold_4[user]['lam'] 
                                Threshold_4[user]['TH2_1'] = alpha*Threshold_4[user]['TH2_1'] - Threshold_4[user]['lam']
                                Threshold_4[user]['TH3_11'] = alpha*Threshold_4[user]['TH3_11'] - Threshold_4[user]['lam']
                                Threshold_4[user]['TH4_111'] = alpha*Threshold_4[user]['TH4_111'] + Threshold_4[user]['lam']
                            else:
                                Threshold_4[user]['TH1'] = alpha*Threshold_4[user]['TH1'] + Threshold_4[user]['ome']
                                Threshold_4[user]['TH2_1'] = alpha*Threshold_4[user]['TH2_1'] + Threshold_4[user]['ome']
                                Threshold_4[user]['TH3_11'] = alpha*Threshold_4[user]['TH3_11'] + Threshold_4[user]['ome']
                                Threshold_4[user]['TH4_111'] = alpha*Threshold_4[user]['TH4_111'] - Threshold_4[user]['ome'] 
                            Threshold_4[user]['TH1'] = np.clip(Threshold_4[user]['TH1'],-Z,Z)
                            Threshold_4[user]['TH2_1'] = np.clip(Threshold_4[user]['TH2_1'],-Z,Z)
                            Threshold_4[user]['TH3_11'] = np.clip(Threshold_4[user]['TH3_11'],-Z,Z)
                            Threshold_4[user]['TH4_111'] = np.clip(Threshold_4[user]['TH4_111'],-Z,Z)
                    else:
                        j3 = '0'
                        if(rand4>= Threshold_4[user]['TH4_110']):
                            j4 = '1'
                            j = j1+j2+j3+j4
                            earn = bernoulli.rvs(signal_path_4[user][j][0])
                            signal_path_4[user][j][1] = signal_path_4[user][j][1] + 1 
                            signal_path_4[user][j][2] = signal_path_4[user][j][2] + earn
                            #priv_4[user]['path'] = j
                            # priv_fuzz[user]['ave_earn'][j] = signal_path_4[user][j][2]/signal_path_4[user][j][1]
                            priv_4[user]['ave_earn'][j] = signal_path_4[user][j][2]/signal_path_4[user][j][1]
                            if(earn > 0):
                                Threshold_4[user]['TH1'] = alpha*Threshold_4[user]['TH1'] - Threshold_4[user]['lam'] 
                                Threshold_4[user]['TH2_1'] = alpha*Threshold_4[user]['TH2_1'] - Threshold_4[user]['lam']
                                Threshold_4[user]['TH3_11'] = alpha*Threshold_4[user]['TH3_11'] + Threshold_4[user]['lam']
                                Threshold_4[user]['TH4_110'] = alpha*Threshold_4[user]['TH4_110'] - Threshold_4[user]['lam']
                            else:
                                Threshold_4[user]['TH1'] = alpha*Threshold_4[user]['TH1'] + Threshold_4[user]['ome']
                                Threshold_4[user]['TH2_1'] = alpha*Threshold_4[user]['TH2_1'] + Threshold_4[user]['ome']
                                Threshold_4[user]['TH3_11'] = alpha*Threshold_4[user]['TH3_11'] - Threshold_4[user]['ome']
                                Threshold_4[user]['TH4_110'] = alpha*Threshold_4[user]['TH4_110'] + Threshold_4[user]['ome'] 
                            Threshold_4[user]['TH1'] = np.clip(Threshold_4[user]['TH1'],-Z,Z)
                            Threshold_4[user]['TH2_1'] = np.clip(Threshold_4[user]['TH2_1'],-Z,Z)
                            Threshold_4[user]['TH3_11'] = np.clip(Threshold_4[user]['TH3_11'],-Z,Z)
                            Threshold_4[user]['TH4_110'] = np.clip(Threshold_4[user]['TH4_110'],-Z,Z)
                        else:
                            j4 = '0'
                            j = j1+j2+j3+j4
                            earn = bernoulli.rvs(signal_path_4[user][j][0])
                            signal_path_4[user][j][1] = signal_path_4[user][j][1] + 1 
                            signal_path_4[user][j][2] = signal_path_4[user][j][2] + earn
                            #priv_4[user]['path'] = j
                            # priv_fuzz[user]['ave_earn'][j] = signal_path_4[user][j][2]/signal_path_4[user][j][1]
                            priv_4[user]['ave_earn'][j] = signal_path_4[user][j][2]/signal_path_4[user][j][1]
                            if(earn > 0):
                                Threshold_4[user]['TH1'] = alpha*Threshold_4[user]['TH1'] - Threshold_4[user]['lam'] 
                                Threshold_4[user]['TH2_1'] = alpha*Threshold_4[user]['TH2_1'] - Threshold_4[user]['lam']
                                Threshold_4[user]['TH3_11'] = alpha*Threshold_4[user]['TH3_11'] + Threshold_4[user]['lam']
                                Threshold_4[user]['TH4_110'] = alpha*Threshold_4[user]['TH4_110'] + Threshold_4[user]['lam']
                            else:
                                Threshold_4[user]['TH1'] = alpha*Threshold_4[user]['TH1'] + Threshold_4[user]['ome']
                                Threshold_4[user]['TH2_1'] = alpha*Threshold_4[user]['TH2_1'] + Threshold_4[user]['ome']
                                Threshold_4[user]['TH3_11'] = alpha*Threshold_4[user]['TH3_11'] - Threshold_4[user]['ome']
                                Threshold_4[user]['TH4_110'] = alpha*Threshold_4[user]['TH4_110'] - Threshold_4[user]['ome'] 
                            Threshold_4[user]['TH1'] = np.clip(Threshold_4[user]['TH1'],-Z,Z)
                            Threshold_4[user]['TH2_1'] = np.clip(Threshold_4[user]['TH2_1'],-Z,Z)
                            Threshold_4[user]['TH3_11'] = np.clip(Threshold_4[user]['TH3_11'],-Z,Z)
                            Threshold_4[user]['TH4_110'] = np.clip(Threshold_4[user]['TH4_110'],-Z,Z)                     
                else:
                    j2 = '0'
                    if(rand3>= Threshold_4[user]['TH3_10']):
                        j3 = '1'
                        if(rand4>= Threshold_4[user]['TH4_101']):
                            j4 = '1'
                            j = j1+j2+j3+j4
                            earn = bernoulli.rvs(signal_path_4[user][j][0])
                            signal_path_4[user][j][1] = signal_path_4[user][j][1] + 1 
                            signal_path_4[user][j][2] = signal_path_4[user][j][2] + earn
                            #priv_4[user]['path'] = j
                            # priv_fuzz[user]['ave_earn'][j] = signal_path_4[user][j][2]/signal_path_4[user][j][1]
                            priv_4[user]['ave_earn'][j] = signal_path_4[user][j][2]/signal_path_4[user][j][1]
                            if(earn > 0):
                                Threshold_4[user]['TH1'] = alpha*Threshold_4[user]['TH1'] - Threshold_4[user]['lam'] 
                                Threshold_4[user]['TH2_1'] = alpha*Threshold_4[user]['TH2_1'] + Threshold_4[user]['lam']
                                Threshold_4[user]['TH3_10'] = alpha*Threshold_4[user]['TH3_10'] - Threshold_4[user]['lam']
                                Threshold_4[user]['TH4_101'] = alpha*Threshold_4[user]['TH4_101'] - Threshold_4[user]['lam']
                            else:
                                Threshold_4[user]['TH1'] = alpha*Threshold_4[user]['TH1'] + Threshold_4[user]['ome']
                                Threshold_4[user]['TH2_1'] = alpha*Threshold_4[user]['TH2_1'] - Threshold_4[user]['ome']
                                Threshold_4[user]['TH3_10'] = alpha*Threshold_4[user]['TH3_10'] + Threshold_4[user]['ome']
                                Threshold_4[user]['TH4_101'] = alpha*Threshold_4[user]['TH4_101'] + Threshold_4[user]['ome'] 
                            Threshold_4[user]['TH1'] = np.clip(Threshold_4[user]['TH1'],-Z,Z)
                            Threshold_4[user]['TH2_1'] = np.clip(Threshold_4[user]['TH2_1'],-Z,Z)
                            Threshold_4[user]['TH3_10'] = np.clip(Threshold_4[user]['TH3_10'],-Z,Z)
                            Threshold_4[user]['TH4_101'] = np.clip(Threshold_4[user]['TH4_101'],-Z,Z)
                        else:
                            j4 = '0'
                            j = j1+j2+j3+j4
                            earn = bernoulli.rvs(signal_path_4[user][j][0])
                            signal_path_4[user][j][1] = signal_path_4[user][j][1] + 1 
                            signal_path_4[user][j][2] = signal_path_4[user][j][2] + earn
                            #priv_4[user]['path'] = j
                            # priv_fuzz[user]['ave_earn'][j] = signal_path_4[user][j][2]/signal_path_4[user][j][1]
                            priv_4[user]['ave_earn'][j] = signal_path_4[user][j][2]/signal_path_4[user][j][1]
                            if(earn > 0):
                                Threshold_4[user]['TH1'] = alpha*Threshold_4[user]['TH1'] - Threshold_4[user]['lam'] 
                                Threshold_4[user]['TH2_1'] = alpha*Threshold_4[user]['TH2_1'] + Threshold_4[user]['lam']
                                Threshold_4[user]['TH3_10'] = alpha*Threshold_4[user]['TH3_10'] - Threshold_4[user]['lam']
                                Threshold_4[user]['TH4_101'] = alpha*Threshold_4[user]['TH4_101'] + Threshold_4[user]['lam']
                            else:
                                Threshold_4[user]['TH1'] = alpha*Threshold_4[user]['TH1'] + Threshold_4[user]['ome']
                                Threshold_4[user]['TH2_1'] = alpha*Threshold_4[user]['TH2_1'] - Threshold_4[user]['ome']
                                Threshold_4[user]['TH3_10'] = alpha*Threshold_4[user]['TH3_10'] + Threshold_4[user]['ome']
                                Threshold_4[user]['TH4_101'] = alpha*Threshold_4[user]['TH4_101'] - Threshold_4[user]['ome'] 
                            Threshold_4[user]['TH1'] = np.clip(Threshold_4[user]['TH1'],-Z,Z)
                            Threshold_4[user]['TH2_1'] = np.clip(Threshold_4[user]['TH2_1'],-Z,Z)
                            Threshold_4[user]['TH3_10'] = np.clip(Threshold_4[user]['TH3_10'],-Z,Z)
                            Threshold_4[user]['TH4_101'] = np.clip(Threshold_4[user]['TH4_101'],-Z,Z)
                    else:
                        j3 = '0'
                        if(rand4>= Threshold_4[user]['TH4_100']):
                            j4 = '1'
                            j = j1+j2+j3+j4
                            earn = bernoulli.rvs(signal_path_4[user][j][0])
                            signal_path_4[user][j][1] = signal_path_4[user][j][1] + 1 
                            signal_path_4[user][j][2] = signal_path_4[user][j][2] + earn
                            #priv_4[user]['path'] = j
                            # priv_fuzz[user]['ave_earn'][j] = signal_path_4[user][j][2]/signal_path_4[user][j][1]
                            priv_4[user]['ave_earn'][j] = signal_path_4[user][j][2]/signal_path_4[user][j][1]
                            if(earn > 0):
                                Threshold_4[user]['TH1'] = alpha*Threshold_4[user]['TH1'] - Threshold_4[user]['lam'] 
                                Threshold_4[user]['TH2_1'] = alpha*Threshold_4[user]['TH2_1'] + Threshold_4[user]['lam']
                                Threshold_4[user]['TH3_10'] = alpha*Threshold_4[user]['TH3_10'] + Threshold_4[user]['lam']
                                Threshold_4[user]['TH4_100'] = alpha*Threshold_4[user]['TH4_100'] - Threshold_4[user]['lam']
                            else:
                                Threshold_4[user]['TH1'] = alpha*Threshold_4[user]['TH1'] + Threshold_4[user]['ome']
                                Threshold_4[user]['TH2_1'] = alpha*Threshold_4[user]['TH2_1'] - Threshold_4[user]['ome']
                                Threshold_4[user]['TH3_10'] = alpha*Threshold_4[user]['TH3_10'] - Threshold_4[user]['ome']
                                Threshold_4[user]['TH4_100'] = alpha*Threshold_4[user]['TH4_100'] + Threshold_4[user]['ome'] 
                            Threshold_4[user]['TH1'] = np.clip(Threshold_4[user]['TH1'],-Z,Z)
                            Threshold_4[user]['TH2_1'] = np.clip(Threshold_4[user]['TH2_1'],-Z,Z)
                            Threshold_4[user]['TH3_10'] = np.clip(Threshold_4[user]['TH3_10'],-Z,Z)
                            Threshold_4[user]['TH4_100'] = np.clip(Threshold_4[user]['TH4_100'],-Z,Z)
                        else:
                            j4 = '0'
                            j = j1+j2+j3+j4
                            earn = bernoulli.rvs(signal_path_4[user][j][0])
                            signal_path_4[user][j][1] = signal_path_4[user][j][1] + 1 
                            signal_path_4[user][j][2] = signal_path_4[user][j][2] + earn
                            #priv_4[user]['path'] = j
                            # priv_fuzz[user]['ave_earn'][j] = signal_path_4[user][j][2]/signal_path_4[user][j][1]
                            priv_4[user]['ave_earn'][j] = signal_path_4[user][j][2]/signal_path_4[user][j][1]
                            if(earn > 0):
                                Threshold_4[user]['TH1'] = alpha*Threshold_4[user]['TH1'] - Threshold_4[user]['lam'] 
                                Threshold_4[user]['TH2_1'] = alpha*Threshold_4[user]['TH2_1'] + Threshold_4[user]['lam']
                                Threshold_4[user]['TH3_10'] = alpha*Threshold_4[user]['TH3_10'] + Threshold_4[user]['lam']
                                Threshold_4[user]['TH4_100'] = alpha*Threshold_4[user]['TH4_100'] + Threshold_4[user]['lam']
                            else:
                                Threshold_4[user]['TH1'] = alpha*Threshold_4[user]['TH1'] + Threshold_4[user]['ome']
                                Threshold_4[user]['TH2_1'] = alpha*Threshold_4[user]['TH2_1'] - Threshold_4[user]['ome']
                                Threshold_4[user]['TH3_10'] = alpha*Threshold_4[user]['TH3_10'] - Threshold_4[user]['ome']
                                Threshold_4[user]['TH4_100'] = alpha*Threshold_4[user]['TH4_100'] - Threshold_4[user]['ome'] 
                            Threshold_4[user]['TH1'] = np.clip(Threshold_4[user]['TH1'],-Z,Z)
                            Threshold_4[user]['TH2_1'] = np.clip(Threshold_4[user]['TH2_1'],-Z,Z)
                            Threshold_4[user]['TH3_10'] = np.clip(Threshold_4[user]['TH3_10'],-Z,Z)
                            Threshold_4[user]['TH4_100'] = np.clip(Threshold_4[user]['TH4_100'],-Z,Z)
            else:
                j1 = '0'
                if(rand2>= Threshold_4[user]['TH2_0']):
                    j2 = '1'
                    if(rand3>= Threshold_4[user]['TH3_01']):
                        j3 = '1'
                        if(rand4>= Threshold_4[user]['TH4_011']):
                            j4 = '1'
                            j = j1+j2+j3+j4
                            earn = bernoulli.rvs(signal_path_4[user][j][0])
                            signal_path_4[user][j][1] = signal_path_4[user][j][1] + 1 
                            signal_path_4[user][j][2] = signal_path_4[user][j][2] + earn
                            #priv_4[user]['path'] = j
                            # priv_fuzz[user]['ave_earn'][j] = signal_path_4[user][j][2]/signal_path_4[user][j][1]
                            priv_4[user]['ave_earn'][j] = signal_path_4[user][j][2]/signal_path_4[user][j][1]
                            if(earn > 0):
                                Threshold_4[user]['TH1'] = alpha*Threshold_4[user]['TH1'] + Threshold_4[user]['lam'] 
                                Threshold_4[user]['TH2_0'] = alpha*Threshold_4[user]['TH2_0'] - Threshold_4[user]['lam']
                                Threshold_4[user]['TH3_01'] = alpha*Threshold_4[user]['TH3_01'] - Threshold_4[user]['lam']
                                Threshold_4[user]['TH4_011'] = alpha*Threshold_4[user]['TH4_011'] - Threshold_4[user]['lam']
                            else:
                                Threshold_4[user]['TH1'] = alpha*Threshold_4[user]['TH1'] - Threshold_4[user]['ome']
                                Threshold_4[user]['TH2_0'] = alpha*Threshold_4[user]['TH2_0'] + Threshold_4[user]['ome']
                                Threshold_4[user]['TH3_01'] = alpha*Threshold_4[user]['TH3_01'] + Threshold_4[user]['ome']
                                Threshold_4[user]['TH4_011'] = alpha*Threshold_4[user]['TH4_011'] + Threshold_4[user]['ome'] 
                            Threshold_4[user]['TH1'] = np.clip(Threshold_4[user]['TH1'],-Z,Z)
                            Threshold_4[user]['TH2_0'] = np.clip(Threshold_4[user]['TH2_0'],-Z,Z)
                            Threshold_4[user]['TH3_01'] = np.clip(Threshold_4[user]['TH3_01'],-Z,Z)
                            Threshold_4[user]['TH4_011'] = np.clip(Threshold_4[user]['TH4_011'],-Z,Z)
                        else:
                            j4 = '0'
                            j = j1+j2+j3+j4
                            earn = bernoulli.rvs(signal_path_4[user][j][0])
                            signal_path_4[user][j][1] = signal_path_4[user][j][1] + 1 
                            signal_path_4[user][j][2] = signal_path_4[user][j][2] + earn
                            #priv_4[user]['path'] = j
                            # priv_fuzz[user]['ave_earn'][j] = signal_path_4[user][j][2]/signal_path_4[user][j][1]
                            priv_4[user]['ave_earn'][j] = signal_path_4[user][j][2]/signal_path_4[user][j][1]
                            if(earn > 0):
                                Threshold_4[user]['TH1'] = alpha*Threshold_4[user]['TH1'] + Threshold_4[user]['lam'] 
                                Threshold_4[user]['TH2_0'] = alpha*Threshold_4[user]['TH2_0'] - Threshold_4[user]['lam']
                                Threshold_4[user]['TH3_01'] = alpha*Threshold_4[user]['TH3_01'] - Threshold_4[user]['lam']
                                Threshold_4[user]['TH4_011'] = alpha*Threshold_4[user]['TH4_011'] + Threshold_4[user]['lam']
                            else:
                                Threshold_4[user]['TH1'] = alpha*Threshold_4[user]['TH1'] - Threshold_4[user]['ome']
                                Threshold_4[user]['TH2_0'] = alpha*Threshold_4[user]['TH2_0'] + Threshold_4[user]['ome']
                                Threshold_4[user]['TH3_01'] = alpha*Threshold_4[user]['TH3_01'] + Threshold_4[user]['ome']
                                Threshold_4[user]['TH4_011'] = alpha*Threshold_4[user]['TH4_011'] - Threshold_4[user]['ome'] 
                            Threshold_4[user]['TH1'] = np.clip(Threshold_4[user]['TH1'],-Z,Z)
                            Threshold_4[user]['TH2_0'] = np.clip(Threshold_4[user]['TH2_0'],-Z,Z)
                            Threshold_4[user]['TH3_01'] = np.clip(Threshold_4[user]['TH3_01'],-Z,Z)
                            Threshold_4[user]['TH4_011'] = np.clip(Threshold_4[user]['TH4_011'],-Z,Z)
                    else:
                        j3 = '0'
                        if(rand4>= Threshold_4[user]['TH4_010']):
                            j4 = '1'
                            j = j1+j2+j3+j4
                            earn = bernoulli.rvs(signal_path_4[user][j][0])
                            signal_path_4[user][j][1] = signal_path_4[user][j][1] + 1 
                            signal_path_4[user][j][2] = signal_path_4[user][j][2] + earn
                            #priv_4[user]['path'] = j
                            # priv_fuzz[user]['ave_earn'][j] = signal_path_4[user][j][2]/signal_path_4[user][j][1]
                            priv_4[user]['ave_earn'][j] = signal_path_4[user][j][2]/signal_path_4[user][j][1]
                            if(earn > 0):
                                Threshold_4[user]['TH1'] = alpha*Threshold_4[user]['TH1'] + Threshold_4[user]['lam'] 
                                Threshold_4[user]['TH2_0'] = alpha*Threshold_4[user]['TH2_0'] - Threshold_4[user]['lam']
                                Threshold_4[user]['TH3_01'] = alpha*Threshold_4[user]['TH3_01'] + Threshold_4[user]['lam']
                                Threshold_4[user]['TH4_010'] = alpha*Threshold_4[user]['TH4_010'] - Threshold_4[user]['lam']
                            else:
                                Threshold_4[user]['TH1'] = alpha*Threshold_4[user]['TH1'] - Threshold_4[user]['ome']
                                Threshold_4[user]['TH2_0'] = alpha*Threshold_4[user]['TH2_0'] + Threshold_4[user]['ome']
                                Threshold_4[user]['TH3_01'] = alpha*Threshold_4[user]['TH3_01'] - Threshold_4[user]['ome']
                                Threshold_4[user]['TH4_010'] = alpha*Threshold_4[user]['TH4_010'] + Threshold_4[user]['ome']
                            Threshold_4[user]['TH1'] = np.clip(Threshold_4[user]['TH1'],-Z,Z)
                            Threshold_4[user]['TH2_0'] = np.clip(Threshold_4[user]['TH2_0'],-Z,Z)
                            Threshold_4[user]['TH3_01'] = np.clip(Threshold_4[user]['TH3_01'],-Z,Z)
                            Threshold_4[user]['TH4_010'] = np.clip(Threshold_4[user]['TH4_010'],-Z,Z)
                        else:
                            j4 = '0'
                            j = j1+j2+j3+j4
                            earn = bernoulli.rvs(signal_path_4[user][j][0])
                            signal_path_4[user][j][1] = signal_path_4[user][j][1] + 1 
                            signal_path_4[user][j][2] = signal_path_4[user][j][2] + earn
                            #priv_4[user]['path'] = j
                            # priv_fuzz[user]['ave_earn'][j] = signal_path_4[user][j][2]/signal_path_4[user][j][1]
                            priv_4[user]['ave_earn'][j] = signal_path_4[user][j][2]/signal_path_4[user][j][1]
                            if(earn > 0):
                                Threshold_4[user]['TH1'] = alpha*Threshold_4[user]['TH1'] + Threshold_4[user]['lam'] 
                                Threshold_4[user]['TH2_0'] = alpha*Threshold_4[user]['TH2_0'] - Threshold_4[user]['lam']
                                Threshold_4[user]['TH3_01'] = alpha*Threshold_4[user]['TH3_01'] + Threshold_4[user]['lam']
                                Threshold_4[user]['TH4_010'] = alpha*Threshold_4[user]['TH4_010'] + Threshold_4[user]['lam']
                            else:
                                Threshold_4[user]['TH1'] = alpha*Threshold_4[user]['TH1'] - Threshold_4[user]['ome']
                                Threshold_4[user]['TH2_0'] = alpha*Threshold_4[user]['TH2_0'] + Threshold_4[user]['ome']
                                Threshold_4[user]['TH3_01'] = alpha*Threshold_4[user]['TH3_01'] - Threshold_4[user]['ome']
                                Threshold_4[user]['TH4_010'] = alpha*Threshold_4[user]['TH4_010'] - Threshold_4[user]['ome']
                            Threshold_4[user]['TH1'] = np.clip(Threshold_4[user]['TH1'],-Z,Z)
                            Threshold_4[user]['TH2_0'] = np.clip(Threshold_4[user]['TH2_0'],-Z,Z)
                            Threshold_4[user]['TH3_01'] = np.clip(Threshold_4[user]['TH3_01'],-Z,Z)
                            Threshold_4[user]['TH4_010'] = np.clip(Threshold_4[user]['TH4_010'],-Z,Z)
                else:
                    j2 = '0'
                    if(rand3>= Threshold_4[user]['TH3_00']):
                        j3 = '1'
                        if(rand4>= Threshold_4[user]['TH4_001']):
                            j4 = '1'
                            j = j1+j2+j3+j4
                            earn = bernoulli.rvs(signal_path_4[user][j][0])
                            signal_path_4[user][j][1] = signal_path_4[user][j][1] + 1 
                            signal_path_4[user][j][2] = signal_path_4[user][j][2] + earn
                            #priv_4[user]['path'] = j
                            # priv_fuzz[user]['ave_earn'][j] = signal_path_4[user][j][2]/signal_path_4[user][j][1]
                            priv_4[user]['ave_earn'][j] = signal_path_4[user][j][2]/signal_path_4[user][j][1]
                            if(earn > 0):
                                Threshold_4[user]['TH1'] = alpha*Threshold_4[user]['TH1'] + Threshold_4[user]['lam'] 
                                Threshold_4[user]['TH2_0'] = alpha*Threshold_4[user]['TH2_0'] + Threshold_4[user]['lam']
                                Threshold_4[user]['TH3_00'] = alpha*Threshold_4[user]['TH3_00'] - Threshold_4[user]['lam']
                                Threshold_4[user]['TH4_001'] = alpha*Threshold_4[user]['TH4_001'] - Threshold_4[user]['lam']
                            else:
                                Threshold_4[user]['TH1'] = alpha*Threshold_4[user]['TH1'] - Threshold_4[user]['ome']
                                Threshold_4[user]['TH2_0'] = alpha*Threshold_4[user]['TH2_0'] - Threshold_4[user]['ome']
                                Threshold_4[user]['TH3_00'] = alpha*Threshold_4[user]['TH3_00'] + Threshold_4[user]['ome']
                                Threshold_4[user]['TH4_001'] = alpha*Threshold_4[user]['TH4_001'] + Threshold_4[user]['ome']
                            Threshold_4[user]['TH1'] = np.clip(Threshold_4[user]['TH1'],-Z,Z)
                            Threshold_4[user]['TH2_0'] = np.clip(Threshold_4[user]['TH2_0'],-Z,Z)
                            Threshold_4[user]['TH3_00'] = np.clip(Threshold_4[user]['TH3_00'],-Z,Z)
                            Threshold_4[user]['TH4_001'] = np.clip(Threshold_4[user]['TH4_001'],-Z,Z)
                        else:
                            j4 = '0'
                            j = j1+j2+j3+j4
                            earn = bernoulli.rvs(signal_path_4[user][j][0])
                            signal_path_4[user][j][1] = signal_path_4[user][j][1] + 1 
                            signal_path_4[user][j][2] = signal_path_4[user][j][2] + earn
                            #priv_4[user]['path'] = j
                            # priv_fuzz[user]['ave_earn'][j] = signal_path_4[user][j][2]/signal_path_4[user][j][1]
                            priv_4[user]['ave_earn'][j] = signal_path_4[user][j][2]/signal_path_4[user][j][1]
                            if(earn > 0):
                                Threshold_4[user]['TH1'] = alpha*Threshold_4[user]['TH1'] + Threshold_4[user]['lam'] 
                                Threshold_4[user]['TH2_0'] = alpha*Threshold_4[user]['TH2_0'] + Threshold_4[user]['lam']
                                Threshold_4[user]['TH3_00'] = alpha*Threshold_4[user]['TH3_00'] - Threshold_4[user]['lam']
                                Threshold_4[user]['TH4_001'] = alpha*Threshold_4[user]['TH4_001'] + Threshold_4[user]['lam']
                            else:
                                Threshold_4[user]['TH1'] = alpha*Threshold_4[user]['TH1'] - Threshold_4[user]['ome']
                                Threshold_4[user]['TH2_0'] = alpha*Threshold_4[user]['TH2_0'] - Threshold_4[user]['ome']
                                Threshold_4[user]['TH3_00'] = alpha*Threshold_4[user]['TH3_00'] + Threshold_4[user]['ome']
                                Threshold_4[user]['TH4_001'] = alpha*Threshold_4[user]['TH4_001'] - Threshold_4[user]['ome']
                            Threshold_4[user]['TH1'] = np.clip(Threshold_4[user]['TH1'],-Z,Z)
                            Threshold_4[user]['TH2_0'] = np.clip(Threshold_4[user]['TH2_0'],-Z,Z)
                            Threshold_4[user]['TH3_00'] = np.clip(Threshold_4[user]['TH3_00'],-Z,Z)
                            Threshold_4[user]['TH4_001'] = np.clip(Threshold_4[user]['TH4_001'],-Z,Z)
                    else:
                        j3 = '0'
                        if(rand4>= Threshold_4[user]['TH4_000']):
                            j4 = '1'
                            j = j1+j2+j3+j4
                            earn = bernoulli.rvs(signal_path_4[user][j][0])
                            signal_path_4[user][j][1] = signal_path_4[user][j][1] + 1 
                            signal_path_4[user][j][2] = signal_path_4[user][j][2] + earn
                            #priv_4[user]['path'] = j
                            # priv_fuzz[user]['ave_earn'][j] = signal_path_4[user][j][2]/signal_path_4[user][j][1]
                            priv_4[user]['ave_earn'][j] = signal_path_4[user][j][2]/signal_path_4[user][j][1]
                            if(earn > 0):
                                Threshold_4[user]['TH1'] = alpha*Threshold_4[user]['TH1'] + Threshold_4[user]['lam'] 
                                Threshold_4[user]['TH2_0'] = alpha*Threshold_4[user]['TH2_0'] + Threshold_4[user]['lam']
                                Threshold_4[user]['TH3_00'] = alpha*Threshold_4[user]['TH3_00'] + Threshold_4[user]['lam']
                                Threshold_4[user]['TH4_000'] = alpha*Threshold_4[user]['TH4_000'] - Threshold_4[user]['lam']
                            else:
                                Threshold_4[user]['TH1'] = alpha*Threshold_4[user]['TH1'] - Threshold_4[user]['ome']
                                Threshold_4[user]['TH2_0'] = alpha*Threshold_4[user]['TH2_0'] - Threshold_4[user]['ome']
                                Threshold_4[user]['TH3_00'] = alpha*Threshold_4[user]['TH3_00'] - Threshold_4[user]['ome']
                                Threshold_4[user]['TH4_000'] = alpha*Threshold_4[user]['TH4_000'] + Threshold_4[user]['ome']
                            Threshold_4[user]['TH1'] = np.clip(Threshold_4[user]['TH1'],-Z,Z)
                            Threshold_4[user]['TH2_0'] = np.clip(Threshold_4[user]['TH2_0'],-Z,Z)
                            Threshold_4[user]['TH3_00'] = np.clip(Threshold_4[user]['TH3_00'],-Z,Z)
                            Threshold_4[user]['TH4_000'] = np.clip(Threshold_4[user]['TH4_000'],-Z,Z)
                        else:
                            j4 = '0'
                            j = j1+j2+j3+j4
                            earn = bernoulli.rvs(signal_path_4[user][j][0])
                            signal_path_4[user][j][1] = signal_path_4[user][j][1] + 1 
                            signal_path_4[user][j][2] = signal_path_4[user][j][2] + earn
                            #priv_4[user]['path'] = j
                            # priv_fuzz[user]['ave_earn'][j] = signal_path_4[user][j][2]/signal_path_4[user][j][1]
                            priv_4[user]['ave_earn'][j] = signal_path_4[user][j][2]/signal_path_4[user][j][1]
                            if(earn > 0):
                                Threshold_4[user]['TH1'] = alpha*Threshold_4[user]['TH1'] + Threshold_4[user]['lam'] 
                                Threshold_4[user]['TH2_0'] = alpha*Threshold_4[user]['TH2_0'] + Threshold_4[user]['lam']
                                Threshold_4[user]['TH3_00'] = alpha*Threshold_4[user]['TH3_00'] + Threshold_4[user]['lam']
                                Threshold_4[user]['TH4_000'] = alpha*Threshold_4[user]['TH4_000'] + Threshold_4[user]['lam']
                            else:
                                Threshold_4[user]['TH1'] = alpha*Threshold_4[user]['TH1'] - Threshold_4[user]['ome']
                                Threshold_4[user]['TH2_0'] = alpha*Threshold_4[user]['TH2_0'] - Threshold_4[user]['ome']
                                Threshold_4[user]['TH3_00'] = alpha*Threshold_4[user]['TH3_00'] - Threshold_4[user]['ome']
                                Threshold_4[user]['TH4_000'] = alpha*Threshold_4[user]['TH4_000'] - Threshold_4[user]['ome']
                            Threshold_4[user]['TH1'] = np.clip(Threshold_4[user]['TH1'],-Z,Z)
                            Threshold_4[user]['TH2_0'] = np.clip(Threshold_4[user]['TH2_0'],-Z,Z)
                            Threshold_4[user]['TH3_00'] = np.clip(Threshold_4[user]['TH3_00'],-Z,Z)
                            Threshold_4[user]['TH4_000'] = np.clip(Threshold_4[user]['TH4_000'],-Z,Z)
            seq_4 = []
            prefs = sorted(priv_4[user]['ave_earn'].items(),key = lambda x:x[1],reverse= True)
            for pref in prefs:
                seq_4.append(pref[0])
            priv_4[user]['pref'] = seq_4


        #随机选出发起者
        initiators = random.sample(users,init_num)
        #进行比较与交换
        
        for initiator in initiators:
            occ_paths_1 = [];occ_paths_2 = [];occ_paths_3 = [];occ_paths_4 = []
            other_users = list(set(users)-set(initiator))

            ##除了发起者外，其余用户所占用的信道，记为occ_paths
            for other_user in other_users:
                occ_paths_1.append(priv_1[other_user]['path'])
            occ_paths_1 = list(set(occ_paths_1))

            for other_user in other_users:
                occ_paths_2.append(priv_2[other_user]['path'])
            occ_paths_2 = list(set(occ_paths_2))

            for other_user in other_users:
                occ_paths_3.append(priv_3[other_user]['path'])
            occ_paths_3 = list(set(occ_paths_3))

            for other_user in other_users:
                occ_paths_4.append(priv_4[other_user]['path'])
            occ_paths_4 = list(set(occ_paths_4))

            ##对偏好进行遍历
            for path in priv_1[initiator]['pref']:

                if(path in occ_paths_1):
                    ##找到目前占用path的用户，将其命名为the_user
                    for other_user in other_users:
                        if(priv_1[other_user]['path'] == path):
                            the_user = other_user
                            break
                        else:
                            continue
                    ##进行双向比较
                    condition_1 = (priv_1[initiator]['ave_earn'][path] >= priv_1[the_user]['ave_earn'][path])
                    # condition_2 = (priv_1[the_user]['ave_earn'][priv_1[initiator]['path']] >= priv_1[the_user]['ave_earn'][path])
                    # condition_3 = (abs(priv_1[initiator]['ave_earn'][path] - priv_1[the_user]['ave_earn'][path]) <= delta)
                    # condition_4 = (abs(priv_1[the_user]['ave_earn'][priv_1[initiator]['path']] - priv_1[the_user]['ave_earn'][path]) <= delta)
                    if(condition_1 ):
                        change = priv_1[initiator]['path']
                        priv_1[initiator]['path'] = path
                        priv_1[the_user]['path'] = change
                        
                        change_time_1 += 1
                        
                        break
                    else:
                        
                        continue                    
                else: 
                    
                    priv_1[initiator]['path'] = path
                    break
            
            for path in priv_2[initiator]['pref']:

                if(path in occ_paths_2):
                    ##找到目前占用path的用户，将其命名为the_user
                    for other_user in other_users:
                        if(priv_2[other_user]['path'] == path):
                            the_user = other_user
                            break
                        else:
                            continue
                    ##进行双向比较
                    condition_1 = (priv_2[initiator]['ave_earn'][path] >= priv_2[the_user]['ave_earn'][path])
                    # condition_2 = (priv_2[the_user]['ave_earn'][priv_2[initiator]['path']] >= priv_2[the_user]['ave_earn'][path])
                    # condition_3 = (abs(priv_2[initiator]['ave_earn'][path] - priv_2[the_user]['ave_earn'][path]) <= delta)
                    # condition_4 = (abs(priv_2[the_user]['ave_earn'][priv_2[initiator]['path']] - priv_2[the_user]['ave_earn'][path]) <= delta)
                    if(condition_1 ):
                        change = priv_2[initiator]['path']
                        priv_2[initiator]['path'] = path
                        priv_2[the_user]['path'] = change
                        
                        change_time_2 += 1
                        
                        break
                    else:
                        
                        continue                    
                else: 
                    
                    priv_2[initiator]['path'] = path
                    
                    break

            for path in priv_3[initiator]['pref']:

                if(path in occ_paths_3):
                    ##找到目前占用path的用户，将其命名为the_user
                    for other_user in other_users:
                        if(priv_3[other_user]['path'] == path):
                            the_user = other_user
                            break
                        else:
                            continue
                    ##进行双向比较
                    condition_1 = (priv_3[initiator]['ave_earn'][path] >= priv_3[the_user]['ave_earn'][path])
                    # condition_2 = (priv_3[the_user]['ave_earn'][priv_3[initiator]['path']] >= priv_3[the_user]['ave_earn'][path])
                    # condition_3 = (abs(priv_3[initiator]['ave_earn'][path] - priv_3[the_user]['ave_earn'][path]) <= delta)
                    # condition_4 = (abs(priv_3[the_user]['ave_earn'][priv_3[initiator]['path']] - priv_3[the_user]['ave_earn'][path]) <= delta)
                    if(condition_1 ):
                        change = priv_3[initiator]['path']
                        priv_3[initiator]['path'] = path
                        priv_3[the_user]['path'] = change
                        
                        change_time_3 += 1
                        
                        break
                    else:
                        
                        continue                    
                else: 
                    
                    priv_3[initiator]['path'] = path
                    
                    break

            for path in priv_4[initiator]['pref']:

                if(path in occ_paths_4):
                    ##找到目前占用path的用户，将其命名为the_user
                    for other_user in other_users:
                        if(priv_4[other_user]['path'] == path):
                            the_user = other_user
                            break
                        else:
                            continue
                    ##进行双向比较
                    condition_1 = (priv_4[initiator]['ave_earn'][path] >= priv_4[the_user]['ave_earn'][path])
                    # condition_2 = (priv_4[the_user]['ave_earn'][priv_4[initiator]['path']] >= priv_4[the_user]['ave_earn'][path])
                    # condition_3 = (abs(priv_4[initiator]['ave_earn'][path] - priv_4[the_user]['ave_earn'][path]) <= delta)
                    # condition_4 = (abs(priv_4[the_user]['ave_earn'][priv_4[initiator]['path']] - priv_4[the_user]['ave_earn'][path]) <= delta)
                    if(condition_1 ):
                        change = priv_4[initiator]['path']
                        priv_4[initiator]['path'] = path
                        priv_4[the_user]['path'] = change
                        
                        change_time_4 += 1
                        break
                    else:
                        
                        continue                    
                else: 
                    
                    priv_4[initiator]['path'] = path
                    
                    break

        change_time_epoch_1 += change_time_1
        for user in users:
            total_earn_epoch_1 += Bit*signal_path_1[user][priv_1[user]['path']][0]
        
        change_time_epoch_2 += change_time_2
        for user in users:
            total_earn_epoch_2 += Bit*signal_path_2[user][priv_2[user]['path']][0]

        change_time_epoch_3 += change_time_3
        for user in users:
            total_earn_epoch_3 += Bit*signal_path_3[user][priv_3[user]['path']][0]

        change_time_epoch_4 += change_time_4
        for user in users:
            total_earn_epoch_4 += Bit*signal_path_4[user][priv_4[user]['path']][0]


    total_earns_1.append(total_earn_epoch_1/epochs)
    change_times_1.append(change_times_1[time]+change_time_epoch_1/epochs)

    total_earns_2.append(total_earn_epoch_2/epochs)
    change_times_2.append(change_times_2[time]+change_time_epoch_2/epochs)

    total_earns_3.append(total_earn_epoch_3/epochs)
    change_times_3.append(change_times_3[time]+change_time_epoch_3/epochs)

    total_earns_4.append(total_earn_epoch_4/epochs)
    change_times_4.append(change_times_4[time]+change_time_epoch_4/epochs)

               
## 最终画图
fig = plt.figure()

## 不同随机数的比较
sub1 = fig.add_subplot(111)
sub1.plot(total_earns_1, c = 'dimgray', marker = 'o',ls = '-', label = 'Quantum',markersize = 1 )
sub1.plot(total_earns_2, c = 'orange', marker = 'o',ls = '-', label = 'Pseudochaos',markersize = 1 )
sub1.plot(total_earns_3, c = 'purple', marker = 'o',ls = '-', label = 'Qauss',markersize = 1 )
sub1.plot(total_earns_4, c = 'g', marker = 'o',ls = '-', label = 'Rand',markersize = 1 )
sub1.set_xlabel('time')
sub1.set_ylabel('Total_earns')


## 显示图像
plt.legend()
plt.show()