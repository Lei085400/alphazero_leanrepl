
import os
import time
import timeit
from multiprocessing import Pool
import multiprocessing as mp
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
from trainer import Trainer
from mcts import State
from mcts import Node
from mcts import MCTS
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context


def list_files(directory):
    filelist = []
    for file in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, file)):
            print(file)
            filelist.append(file)
    return filelist



def long_time_task(file):
    lean_workdir = "/home2/wanglei/Project" # Lean工程的根目录
    lean_file = "testfolder/succ/" + file  # 待证明定理的Lean文件
   
    print("证明定理为:{}".format(file))
    lean = Lean4Gym(lean_workdir, lean_file)
    try:
        state = lean.getInitState()
    except:
        print("状态异常")
        return False
    init_state = State(state)
    node = Node()
    node.set_state(init_state)
    
    start = timeit.default_timer()

    print("开始搜索策略")
    mcts = MCTS(node, policyModel, valueModel, args, device)
    flag = mcts.runmcts(lean, time_out)

    end = timeit.default_timer()
    count += 1
    if(flag == True):
        succcount += 1
        # succ.append(state.tacticState)
        succ_name.append(format(file))
        
    print ("所用时间：{}".format(str(end-start)))
    print("第{}个定理".format(str(count)))
    print("已成功证明{}条定理".format(str(succcount)))
    
    return flag


if __name__ == '__main__':
    mp.set_start_method('spawn')
    
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu') 

    args = {
        
        'batch_size': 4,
        'numIters': 10,                                # Total number of training iterations
        'num_simulations': 100,                         # Total number of MCTS simulations to run when deciding on a move to play
        'numEps': 20,                                  # Number of full games (episodes) to run during each iteration
        'numItersForTrainExamplesHistory': 20,
        'epochs': 10,                                    # Number of epochs of training per iteration
        'checkpoint_path': 'latest.pth',                 # location to save latest set of weights
        'TACRIC_NUMBER': 8,
        'feature_size':100
        # 'MAX_ROUND_NUMBER' : 10
    }


    state_list = []
    lean_list = []
    count = 0
    succcount = 0
    succ = []
    succ_name = []


    feature_size = args['feature_size']  # 特征向量的size
    time_out = 120
 
    policyModel = policy_model(feature_size*2, device).to(device)
    valueModel = value_model(feature_size, device).to(device)
    print("hello,开始了！！")


    checkpoint_policy = torch.load("/home2/wanglei/Project/alphazero_leanrepl/policy_model")
    policyModel.load_state_dict(checkpoint_policy['state_dict'])

    checkpoint_value = torch.load("/home2/wanglei/Project/alphazero_leanrepl/value_model")
    valueModel.load_state_dict(checkpoint_value['state_dict'])
    
    print(mp.cpu_count())
    pool = Pool(processes=mp.cpu_count() // 3)
    
    #待证明策略：
    lean_dir = "/home2/wanglei/Project/testfolder/succ"
    file_list = list_files(lean_dir)

    
    t1 = timeit.default_timer()
    flag = pool.map(long_time_task, file_list)
    t2 = timeit.default_timer()
    print(t2 - t1)
    pool.close()
    pool.join()

    
    print("成功总数：{}".format(str(count)))
    # print("成功定理有：")
    # print(succ)
    F = open(r'/home2/wanglei/Project/alphazero_leanrepl/0706.txt','w')
    for i in succ_name:
        F.write(str(i)+'\n')
    F.close() 