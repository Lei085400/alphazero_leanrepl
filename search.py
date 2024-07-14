
import os
import time
import timeit
from multiprocessing import Pool
import torch
import pickle
import subprocess
import select
import json
from Lean4Gym import Lean4Gym, ProofState
# print(torch.__version__)
from torch.nn.parallel import DataParallel  
# torch.cuda.set_device(2)

# import lean_dojo
# from lean_dojo import *
from model import policy_model
from model import value_model
from mcts import State
from mcts import Node
from mcts import MCTS
# from normsearch import search
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context



device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu') 


args = {
    
    'batch_size': 4,
    'numIters': 10,                                # Total number of training iterations
    'num_simulations': 100,                         # Total number of MCTS simulations to run when deciding on a move to play
    'numEps': 20,                                  # Number of full games (episodes) to run during each iteration
    'numItersForTrainExamplesHistory': 20,
    'epochs': 10,                                    # Number of epochs of training per iteration
    'checkpoint_path': 'latest.pth',                 # location to save latest set of weights
    'TACRIC_NUMBER': 16,
    'feature_size':100
    # 'MAX_ROUND_NUMBER' : 10
}


state_list = []
lean_list = []

feature_size = args['feature_size']  # 特征向量的size
time_out = 360


device_ids = list(range(torch.cuda.device_count()))  
policyModel = policy_model(feature_size*2, device).to(device)
valueModel = value_model(feature_size, device).to(device)
print("hello,开始了！！")




# #待证明策略：
# lean_workdir = "/home2/wanglei/Project" # Lean工程的根目录
# lean_file = "demo.lean"   # 待证明定理的Lean文件
# lean = Lean4Gym(lean_workdir, lean_file)
# state = lean.getInitState()
# init_state = State(state)
# node = Node()
# node.set_state(init_state)

# start = timeit.default_timer()
# print("第一次搜索策略")
# mcts = MCTS(node, policyModel, valueModel, args, device)
# node = mcts.runmcts(lean,time_out)
# end = timeit.default_timer()
# print ("第一次时间：{}".format(str(end-start)))


checkpoint_policy = torch.load("/home/wanglei/AAAI/lean_ATG/leanproject/alphazero_leanrepl/policy_model")
policyModel.load_state_dict(checkpoint_policy['state_dict'])

checkpoint_value = torch.load("/home/wanglei/AAAI/lean_ATG/leanproject/alphazero_leanrepl/value_model")
valueModel.load_state_dict(checkpoint_value['state_dict'])

def list_files(directory):
    filelist = []
    for i, file in enumerate(os.listdir(directory)):
        if(i < 140):
            continue
        if os.path.isfile(os.path.join(directory, file)):
            print(file)
            filelist.append(file)
    return filelist

count = 0
succ = []
succ_name = []
 

# #待证明策略：
# lean_dir = "/home/wanglei/AAAI/lean_ATG/leanproject/testfolder/lean_theorems_with_options"
# # lean_dir = "/home2/wanglei/Project/testfolder"
# file_list = list_files(lean_dir)
# # print(len(file_list))
# Fi = open(r'/home/wanglei/AAAI/lean_ATG/leanproject/alphazero_leanrepl/file_list.txt','w')
# for i in file_list:
#     Fi.write(str(i)+'\n')
# Fi.close() 

file_list = []
with open('example.txt', 'r') as file: 
    lines = file.readlines() 
    for line in lines:
        line = ''.join(line).strip('\n')
        file_list.append(line)
        print(line)

lean_workdir = "/home/wanglei/AAAI/lean_ATG/leanproject" # Lean工程的根目录
for i, file in enumerate(file_list):
    print("============================================")
    lean_file = "testfolder/lean_theorems_with_options/" + file  # 待证明定理的Lean文件
   
    print("证明定理为:{}".format(file))
    lean = Lean4Gym(lean_workdir, lean_file)
    try:
        state = lean.getInitState()
    except:
        print("状态异常")
        continue
    init_state = State(state)
    node = Node()
    node.set_state(init_state)


    start = timeit.default_timer()

    print("开始搜索策略")
    mcts = MCTS(node, policyModel, valueModel, args, device)
    flag = mcts.runmcts(lean, time_out)
    # flag = search(state, lean)

    end = timeit.default_timer()
    
    if(flag == True):
        count += 1
        # succ.append(state.tacticState)
        succ_name.append(format(file))
        FF = open(r'/home/wanglei/AAAI/lean_ATG/leanproject/alphazero_leanrepl/search_test.txt','a')
        FF.write(file +'\n')
        FF.close() 
        
    print ("所用时间：{}".format(str(end-start)))
    print("第{}个定理".format(str(i)))
    print("已成功证明{}条定理".format(str(count)))
    
print("成功总数：{}".format(str(count)))
print("通过比例：{}".format(str(count/len(file_list))))
# print("成功定理有：")
# print(succ)
F = open(r'/home/wanglei/AAAI/lean_ATG/leanproject/alphazero_leanrepl/output.txt','w')
for i in succ_name:
    F.write(str(i)+'\n')
F.close() 