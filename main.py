
import os

import torch
import pickle
import subprocess

import timeit
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
from trainer import Trainer
from mcts import State
from mcts import Node
from mcts import MCTS
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context



device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu') 


args = {
    'batch_size':64,
    'numIters': 10,                                # Total number of training iterations
    'num_simulations': 100,                         # Total number of MCTS simulations to run when deciding on a move to play
    'numEps': 50,                                  # Number of full games (episodes) to run during each iteration
    'numItersForTrainExamplesHistory': 20,
    'epochs': 100,                                    # Number of epochs of training per iteration
    'checkpoint_path': 'latest.pth',                 # location to save latest set of weights
    'TACRIC_NUMBER': 12,
    'feature_size':100
    # 'MAX_ROUND_NUMBER' : 10
}


state_list = []
lean_list = []

feature_size = args['feature_size']  # 特征向量的size
time_out = 600


device_ids = list(range(torch.cuda.device_count()))  
policyModel = policy_model(feature_size*2, device).to(device)
valueModel = value_model(feature_size, device).to(device)
print("hello,开始了！！")

def list_files(directory):
    filelist = []
    for file in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, file)):
            print(file)
            filelist.append(file)
    return filelist

##################################################################
# lean_workdir = "/home2/wanglei/Project" # Lean工程的根目录
# lean_file = "example_separate/choose_eq_choose_sub_add.lean"   # 待证明定理的Lean文件
# print("lean_workdir:", lean_workdir)
# print("lean_file   :", lean_file)

# lean = Lean4Gym(lean_workdir, lean_file)
# state = lean.getInitState()

# init_state = State(state)

# state_list.append(init_state)
# lean_list.append(lean)
##################################################################
#待证明策略：
lean_dir = "/home2/wanglei/Project/testfolder/succ"
file_list = list_files(lean_dir)
# print(len(file_list))

lean_workdir = "/home2/wanglei/Project" # Lean工程的根目录
for i, file in enumerate(file_list):
    print("============================================")
    if(i>50):
        break
    lean_file = "testfolder/succ/" + file  # 待证明定理的Lean文件
   
    print("证明定理为:{}".format(file))
    lean = Lean4Gym(lean_workdir, lean_file)
    try:
        state = lean.getInitState()
    except:
        print("状态异常")
        continue
    init_state = State(state)
    state_list.append(init_state)
    lean_list.append(lean)

# lean_workdir = "/home2/wanglei/Project" # Lean工程的根目录
# lean_file = "parse/lt_eq_le_sub.lean"   # 待证明定理的Lean文件
# print("lean_workdir:", lean_workdir)
# print("lean_file   :", lean_file)

# lean = Lean4Gym(lean_workdir, lean_file)
# state = lean.getInitState()

# init_state = State(state)

# state_list.append(init_state)
# lean_list.append(lean)

# lean_workdir = "/home2/wanglei/Project" # Lean工程的根目录
# lean_file = "example_separate/mul_two_pow_add_eq_mul_pow.lean"   # 待证明定理的Lean文件
# print("lean_workdir:", lean_workdir)
# print("lean_file   :", lean_file)

# lean = Lean4Gym(lean_workdir, lean_file)
# state = lean.getInitState()

# init_state = State(state)

# state_list.append(init_state)
# lean_list.append(lean)


# ###############################################################

# lean_workdir = "/home2/wanglei/Project" # Lean工程的根目录
# lean_file = "example_separate/Ico_simpn.lean"   # 待证明定理的Lean文件
# print("lean_workdir:", lean_workdir)
# print("lean_file   :", lean_file)

# lean = Lean4Gym(lean_workdir, lean_file)
# state = lean.getInitState()

# init_state = State(state)

# state_list.append(init_state)
# lean_list.append(lean)

# #####################################################################


# lean_workdir = "/home2/wanglei/Project" # Lean工程的根目录
# lean_file = "example_separate/mul_sum_choose_sub_choose.lean"   # 待证明定理的Lean文件
# print("lean_workdir:", lean_workdir)
# print("lean_file   :", lean_file)

# lean = Lean4Gym(lean_workdir, lean_file)
# state = lean.getInitState()

# init_state = State(state)

# state_list.append(init_state)
# lean_list.append(lean)

# #####################################################################


# lean_workdir = "/home2/wanglei/Project" # Lean工程的根目录
# lean_file = "example_separate/mul_two_div_mul.lean"   # 待证明定理的Lean文件
# print("lean_workdir:", lean_workdir)
# print("lean_file   :", lean_file)

# lean = Lean4Gym(lean_workdir, lean_file)
# state = lean.getInitState()

# init_state = State(state)

# state_list.append(init_state)
# lean_list.append(lean)

# #####################################################################

# lean_workdir = "/home2/wanglei/Project" # Lean工程的根目录
# lean_file = "example_separate/choose_le_sum.lean"   # 待证明定理的Lean文件
# print("lean_workdir:", lean_workdir)
# print("lean_file   :", lean_file)

# lean = Lean4Gym(lean_workdir, lean_file)
# state = lean.getInitState()

# init_state = State(state)

# state_list.append(init_state)
# lean_list.append(lean)

# #####################################################################

# lean_workdir = "/home2/wanglei/Project" # Lean工程的根目录
# lean_file = "example_separate/sum_mul_choose_eq_mul_sub.lean"   # 待证明定理的Lean文件
# print("lean_workdir:", lean_workdir)
# print("lean_file   :", lean_file)

# lean = Lean4Gym(lean_workdir, lean_file)
# state = lean.getInitState()

# init_state = State(state)

# state_list.append(init_state)
# lean_list.append(lean)
# # #####################################################################

# lean_workdir = "/home2/wanglei/Project" # Lean工程的根目录
# lean_file = "example_separate/Ico_div.lean"   # 待证明定理的Lean文件
# print("lean_workdir:", lean_workdir)
# print("lean_file   :", lean_file)

# lean = Lean4Gym(lean_workdir, lean_file)
# state = lean.getInitState()

# init_state = State(state)

# state_list.append(init_state)
# lean_list.append(lean)
# #####################################################################
# lean_workdir = "/home2/wanglei/Project" # Lean工程的根目录
# lean_file = "example_separate/Ico_div.lean"   # 待证明定理的Lean文件
# print("lean_workdir:", lean_workdir)
# print("lean_file   :", lean_file)

# lean = Lean4Gym(lean_workdir, lean_file)
# state = lean.getInitState()

# init_state = State(state)

# state_list.append(init_state)
# lean_list.append(lean)
#####################################################################



#待证明策略：
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


trainer = Trainer(policyModel, valueModel, args, device)
print("马上开始训练")
trainer.learn(state_list, lean_list)

# trainer.train()

# ===================================================================
# start = timeit.default_timer()

# print("开始搜索策略")
# mcts = MCTS(node, policyModel, valueModel, args, device)
# node = mcts.runmcts(lean, time_out)
# 2
# end = timeit.default_timer()

# print ("第二次时间：{}".format(str(end-start)))