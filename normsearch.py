#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Paper from http://pubs.doc.ic.ac.uk/survey-mcts-methods/survey-mcts-methods.pdf .
import os
import sys
import math
import random
import numpy as np
# from lean_dojo import *
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# torch.cuda.set_device(2)
import timeit
from Lean4Gym import Lean4Gym, ProofState
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import json
import heapq
import transformers
import subprocess
import vllm
import time
import tqdm
from datetime import datetime
from tqdm import tqdm, trange
from pathlib import Path
from Lean4Gym import *
# from lean_dojo import *
import traceback
# TACRIC_NUMBER = 8
MAX_ROUND_NUMBER = 10

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2" 

tokenizer1 = AutoTokenizer.from_pretrained("/home/wanglei/AAAI/lean_ATG/leanproject/leandojo-lean4-tacgen-byt5-small")       # Or "lean3" -> "lean4"
model = AutoModelForSeq2SeqLM.from_pretrained("/home/wanglei/AAAI/lean_ATG/leanproject/leandojo-lean4-tacgen-byt5-small")   # Or "lean3" -> "lean4"

model_name_or_path = "/home/wanglei/AAAI/lean_ATG/model/pythia2.8b_choose"

model = vllm.LLM(
    model=model_name_or_path,
    tensor_parallel_size=1,
    trust_remote_code=True,
    gpu_memory_utilization=0.8,
    dtype='float16'
)

tokenizer = transformers.GPTNeoXTokenizerFast.from_pretrained(model_name_or_path)

# 模型的prompt输入
def _prompt_proofstep(ts):
    prompt = f"[GOAL]{ts}[PROOFSTEP]"
    if(len(prompt)>2048):
      prompt = prompt[:2048]
    return prompt

# 去重排序
def _unique_sorted(texts, scores):
    texts_ = []
    scores_ = []
    for t, s in sorted(zip(texts, scores), key=lambda x: -x[1]):
        if t not in texts_:
            texts_.append(t)
            scores_.append(s)
    return texts_, scores_

# 模型输入产生输出
def generate_vllm(prompt, model, tokenizer, temperatures, num_samples, stop, max_tokens=1024):
    texts, scores = [], []
    for temperature in temperatures:
        params = vllm.SamplingParams(
            n=num_samples,
            temperature=temperature,
            use_beam_search=temperature==0.0,
            max_tokens=max_tokens,
            stop=stop,
            logprobs=0,
            top_k=-1
        )
        outputs = model.generate([prompt], params, use_tqdm=False)
        
        
        if len(outputs) == 0:
            return [], []
        for output in outputs[0].outputs:
            text = output.text.replace(tokenizer.eos_token, '') 
            score = output.cumulative_logprob/max(len(output.token_ids), 1)
            texts.append(text)
            scores.append(score)

    texts, scores = _unique_sorted(texts, scores)
    return texts, scores
  

def encode_state(state, feature_size):
    # print("!!!!!!!!!!!!!!!!!!!!!!!")
    # encode_tactic_ = [ord(char) for char in state]
    # print(encode_tactic_)
    # print("!!!!!!!!!!!!!!!!!!!!!!!")
    encode_state = tokenizer1.encode(str(state))
    if(len(encode_state)<=feature_size):
        encode_state += [0]*(feature_size-len(encode_state))  #list
    else:
        encode_state = encode_state[:feature_size]
    # print("encode")
    # print(encode_state)
    return encode_state
 
def encode_tactic(tactic, feature_size):
    encode_tactic = tokenizer1.encode(str(tactic))
    if(len(encode_tactic)<=feature_size):
        encode_tactic += [0]*(feature_size-len(encode_tactic))
    else:
        encode_tactic = encode_tactic[:feature_size]
    return encode_tactic

# def tactic_generator(state):
#     init_state = state.pp
#     tokenized_state = tokenizer(init_state, return_tensors="pt")

#     # Generate multiple tactics via beam search.
#     tactic_candidates_ids = model.generate(
#         tokenized_state.input_ids,
#         max_length=1024,
#         num_beams=4,
#         length_penalty=0.0,
#         do_sample=False,
#         num_return_sequences=4,
#         early_stopping=False,
#     )
#     tactic_candidates = tokenizer.batch_decode(
#         tactic_candidates_ids, skip_special_tokens=True
#     )
#     return tactic_candidates

def tactic_generator(state):
  state = state.getTacticState()
  tactic_candidates, scores = generate_vllm(_prompt_proofstep(state), model, tokenizer, 
                              temperatures=[0], num_samples=16, stop=tokenizer.eos_token)

  # print(tactic_candidates)
  # tactic_candidates = [ 'have choose_eq_choose_sub_add :  choose n k = choose (n - 1 + 1) (k - 1 + 1)  := by rw[Nat.sub_add_cancel h1, Nat.sub_add_cancel h2]',
  #                     'rw[choose_eq_choose_sub_add,add_comm (choose (n - 1) k) (choose (n - 1) (k - 1))]',
  #                     'have choose_sub_eq_choose_sub_add : choose (n - 1) k = choose (n - 1) (k - 1 + 1) := by rw[Nat.sub_add_cancel h2]',
  #                     'rw[choose_sub_eq_choose_sub_add, choose_succ_succ]',
  #                     'simp [*, choose]' ,
  #                     'rw [add_assoc]',
  #                     'rw[two_mul]',
  #                     'simp',
  #                     'rw[add_comm]',]
  # tactic_candidates = [ 'have choose_eq_choose_sub_add :  choose n k = choose (n - 1 + 1) (k - 1 + 1)  := by rw[Nat.sub_add_cancel h1, Nat.sub_add_cancel h2]',
  #                       'rw[choose_eq_choose_sub_add,add_comm (choose (n - 1) k) (choose (n - 1) (k - 1))]',
  #                       'have choose_sub_eq_choose_sub_add : choose (n - 1) k = choose (n - 1) (k - 1 + 1) := by rw[Nat.sub_add_cancel h2]',
  #                       'rw[choose_sub_eq_choose_sub_add, choose_succ_succ]',
  #                       'simp [*, choose]' ,
  #                       'rw [add_assoc]',
  #                       'rw[two_mul]',
  #                       'simp',
  #                       'rw[add_comm]',
  #                       'rw[sub_eq_neg_add]',
  #                       'rw[neg_sub]',
  #                       'rw[← add_assoc]',
  #                       'congr 1',
  #                       'rw[Nat.add_div_of_dvd_left]',
  #                       'exact two_mod_two_pow hn',
  #                       'rw[Nat.sub_add_cancel h1]',
  #                       'rw[sum_range_succ]',
  #                       'rw[choose_mul]',
  #                       'rw[Nat.sub_add_cancel h2]',
  #                       'rw[sum_Ico_eq_sum_range]',
  #                       'rw[add_mul]',
  #                       'rw[mul_choose_eq_mul_choose hn]',
  #                       'rw[range_eq_Ico]',
  #                       'rw[← add_mul]',
  #                       'rw[mul_choose_two_pow hn]',
  #                       'rw [sum_Ico_succ_top hn]',
  #                       'rw[div_eq_mul_one_div]',
  #                       'rw[range_sub_choose_add_sum]',
  #                       'rw[pow_add]',
  #                       'norm_num',
  #                     ]
  return tactic_candidates


class State(object):
  
  def __init__(self,state):
    self.tac = None  #记录当前状态是通过哪个策略得到的
    self.state = state
    self.tactic_candidates = None

  def is_terminal(self):  ############# 更改为 证明是否完成（证明成功or失败）
    return self.state.isFinish() or self.state.error is not None

  def compute_reward(self):   ############# 证明成功为1，失败为0
    if (self.state.isFinish()):
      return 1
    elif (self.state.error is not None):
      return -1
    else:
      return None
  
  def proof(self,state,tac,lean):
      result = lean.run_tactic(state, [tac])
      # print("当前状态为{}".format(state))
      # print("策略为{}".format(tac))
      
      # if(result.error is not None):  ## 证明失败，terminal = 1 ， reward = -1
      #     # print("证明失败")
      #     return -1
      # else:
      #   if(result.isFinish()): #证明成功
      #     return 1
      #   else:
      return result
          # if(result == ProofGivenUp()):
          #     print(result)
          #     # print(dojo.is_successful)
          #     # print("证明放弃，terminal = 1 ， reward = -1")
          #     return -1
          # else:
          #     try:
          #         if(result.pp is not None):
          #             # print(result)
          #             # # print(dojo.is_successful)
          #             # print("证明未完成，terminal = 0 ， reward = 0")
          #             return result
          #     except Exception as ex:
          #         if(dojo.is_successful == True): ## 证明成功
          #             # print(result)
          #             # # print(dojo.is_successful)
          #             # print("证明成功，terminal = 1 ， reward = 1")
          #             return 1

  def get_next_state_with_random_choice(self,lean,index):  ############# 根据当前state输入大模型，获取策略list后，随机选择其中一个策略，返回执行该随机策略后的状态
    if(self.tactic_candidates is None):
      self.tactic_candidates = tactic_generator(self.state)
      # print(self.tactic_candidates)
    tactic_candidates = self.tactic_candidates
    # self.state.print()
    # print(tactic_candidates)
    # random_choice = random.choices([choice for choice in tactic_candidates],k=1)
    try:
      random_choice = tactic_candidates[index]
    except:
      self.state.error = "llm error"
      return self
    next_state = self.proof(self.state,random_choice,lean)
    next_state = State(next_state)
    next_state.tac = random_choice
    # next_state = proof(self.state,random_choice)
    # next_node = Node()
    # next_node.set_state(next_state)
    # next_node.tac = random_choice
    return next_state

  
class Node(object):
  """
  蒙特卡罗树搜索的树结构的Node，包含了父节点和直接点等信息，还有用于计算UCB的遍历次数和quality值，还有游戏选择这个Node的State。
  """

  def __init__(self):
    self.parent = None
    self.children = []
    self.prob = 0
    self.puct = 0
    self.visit_times = 0
    self.quality_value = 0.0
    self.flag = 0 #记录该节点所有子节点是否可行，都不可行则为1

    self.state = None
    
    self.depth = 0 

  def set_state(self, state):
    self.state = state

  def get_state(self):  
    return self.state

  def set_parent(self, parent):  
    self.parent = parent

  def get_children(self):  
    return self.children

  def get_visit_times(self):  
    return self.visit_times

  def visit_times_add_one(self):  
    self.visit_times += 1

  def get_quality_value(self): 
    return self.quality_value

  def quality_value_add_n(self, n):  
    self.quality_value += n

  def is_all_expand(self,TACRIC_NUMBER): #### 判断该状态下，是否所有的list中的策略都尝试过了
    return len(self.children) == TACRIC_NUMBER

  def add_child(self, sub_node):
    sub_node.set_parent(self)
    self.children.append(sub_node)
  
  def select_action(self):
    """
    Select action according to the visit count distribution and the temperature.
    """
    visit_counts = np.array([child.visit_times for child in self.children])
    actions = [action for action in self.children.state.tac]
    action = actions[np.argmafx(visit_counts)]
    return action


  def __repr__(self):
    return "Node: {}, Q/N: {}/{}, state: {}".format(
        hash(self), self.quality_value, self.visit_times, self.state)


class MCTS:

    def __init__(self, node, policy_model, value_model, args, device):
        self.node = node
        self.policy_model = policy_model
        self.value_model = value_model
        self.args = args
        self.device = device

    def tree_policy(self, node, lean, is_exploration ):
      """
      蒙特卡罗树搜索的Selection和Expansion阶段，传入当前需要开始搜索的节点（例如根节点），根据exploration/exploitation算法返回最好的需要expend的节点，注意如果节点是叶子结点直接返回。

      基本策略是先找当前未选择过的子节点，如果有多个则随机选。如果都选择过就找权衡过exploration/exploitation的UCB值最大的，如果UCB值相等则随机选。
      """

      # Check if the current node is the leaf node
      while node.state.is_terminal() == False:
        
        if node.is_all_expand(self.args['TACRIC_NUMBER']):
          # print(node.state.state.tacticState)
          best_node = self.best_child(node, is_exploration,False)
          if(best_node is None):
            # print("该节点的子节点的所有策略都无效{}".format(node.state.state.tacticState))
            # best_node = self.best_child(node, is_exploration,True)
            # for sub_node in node.get_children():
            #   if(sub_node is not None):
            #     print("子节点策略为：{}".format(sub_node.state.tac))
            #     sub_node.state.state.print()
            node.flag = 1 
            if(node.parent is not None):
              node = node.parent
            else:
              print("目标无解")
              node.flag = -2
              return node
          else:
            node = best_node
        else:
          # Return the new sub node
          sub_node = self.expand(node,lean)
          return sub_node

      # Return the leaf node
      return node


    def default_policy(self,node):
      """
      蒙特卡罗树搜索的Simulation阶段，输入一个需要expand的节点，随机操作后创建新的节点，返回新增节点的reward。注意输入的节点应该不是子节点，而且是有未执行的Action可以expend的。

      基本策略是随机选择Action。
      """

      # # Get the state of the game
      # current_state = node.get_state()

      # Run until the game over

      state = node.state

      while state.is_terminal() == False:

        # Pick one random action to play and get next state
        state = state.get_next_state_with_random_choice()
        
      
      final_state_reward = state.compute_reward()
      return final_state_reward


    def expand(self,node, lean):
      """
      输入一个节点，在该节点上拓展一个新的节点，使用random方法执行Action，返回新增的节点。注意，需要保证新增的节点与其他节点Action不同。
      """

      # tried_sub_node_states = [     # 统计node已经展开的所有子节点
      #     sub_node.get_state().state.tacticState for sub_node in node.get_children()
      # ]
      
      # tried_sub_node_tacs = [     # 统计node已经展开的所有子节点
      #     sub_node.get_state().tac for sub_node in node.get_children()
      # ]

      new_state = node.state.get_next_state_with_random_choice(lean, len(node.children))   # 根据当前node状态随机采取action，获得执行后的新状态

      # Check until get the new state which has the different action from others
      # while new_state.state.tacticState in tried_sub_node_states and new_state.tac in tried_sub_node_tacs:  # 判断新状态是否已经被expand，若已经被expand，则重新对node随机采取action，获得新状态
      #   new_state = node.state.get_next_state_with_random_choice(lean)   # 根据当前node状态随机采取action，获得执行后的新状态
      
      new_node = Node()
      new_node.set_state(new_state)
      new_node.depth = node.depth + 1
      #########################
      encodestate = encode_state(node.state.state.tacticState, self.args['feature_size'])
      encodetactic = encode_tactic(new_state.tac, self.args['feature_size'])
      input_policy = encodestate + encodetactic
      input_policy = torch.FloatTensor(np.array(input_policy).astype(np.float64)).to(self.device)
      new_node.prob = float(self.policy_model(input_policy))  # 返回的应该不是值，而是数组？
      #########################
      node.add_child(new_node)

      return new_node


    def best_child(self, node, is_exploration,f):
      """
      使用UCB算法，权衡exploration和exploitation后选择得分最高的子节点，注意如果是预测阶段直接选择当前Q值得分最高的。
      """

      # TODO: Use the min float value
      best_score = -sys.maxsize
      best_sub_node = None

      # Travel all sub nodes to find the best one
      # print("当前节点为{}".format(node.state.state.tacticState))
      for sub_node in node.get_children():
        # print("当前孩子节点为{}".format(sub_node.state.state.tacticState))
        if(sub_node.state.state.error is not None):
          # print("该子节点报错")
          # sub_node.state.state.print()
          # print("该子节点报错")
          continue
        if(sub_node.flag==1):
          # print("该子节点的所有策略都无效{}".format(sub_node.state.state.tacticState))
          continue
        # print(sub_node.state.state.tacticState)
        # Ignore exploration for inference
        if is_exploration:
          C = 1 / math.sqrt(2.0)
          # C = 1
          # C = 2
        else:
          C = 1 / math.sqrt(2.0)

        # UCB = quality / times + C * sqrt(2 * ln(total_times) / times)
        left = sub_node.get_quality_value() / sub_node.get_visit_times()
        right = math.sqrt(node.get_visit_times()) / (sub_node.get_visit_times()+1)
        Puct_score = left + C * sub_node.prob * math.sqrt(right)
        sub_node.puct = Puct_score

        if Puct_score > best_score:
          best_sub_node = sub_node
          best_score = Puct_score
      # print("best:{}".format(best_sub_node.state.state.tacticState))
      return best_sub_node


    def backup(self, node, reward):
      """
      蒙特卡洛树搜索的Backpropagation阶段，输入前面获取需要expend的节点和新执行Action的reward，反馈给expend节点和上游所有节点并更新对应数据。
      """

      # Update util the root node
      while node != None:
        # Update the visit times
        node.visit_times_add_one()

        # Update the quality value
        node.quality_value_add_n(reward)

        # Change the node to the parent node
        node = node.parent


    def run(self,lean):
      """
      实现蒙特卡洛树搜索算法，传入一个根节点，在有限的时间内根据之前已经探索过的树结构expand新节点和更新数据，然后返回只要exploitation最高的子节点。

      蒙特卡洛树搜索包含四个步骤，Selection、Expansion、Simulation、Backpropagation。
      前两步使用tree policy找到值得探索的节点。
      第三步使用default policy也就是在选中的节点上随机算法选一个子节点并计算reward。
      最后一步使用backup也就是把reward更新到所有经过的选中节点的节点上。

      进行预测时，只需要根据Q值选择exploitation最大的节点即可，找到下一个最优的节点。
      """
      node =  self.node
      computation_budget = 20

      # Run as much as possible under the computation budget
      for i in range(computation_budget):

        # 1. Find the best node to expand
        # print("mcts到第{}次，node为：{}".format(i,node.state))
        expand_node = self.tree_policy(node, lean, True)
        if(expand_node.flag == -2):
            return node
        # if(expand_node.state.state == 1):
        #   print("成功策略：")
        #   path = []
        #   state_list = []
        #   while expand_node.state.tac is not None:
        #       path.append(expand_node.state.tac)
        #       state_list.append(expand_node.state.state)
        #       expand_node = expand_node.parent
        #   path.reverse()
        #   state_list.append(node.state.state)
        #   state_list.reverse()
        #   print(path)
        #   print(state_list)
        #   return expand_node

        # 2. Random run to add node and get reward
        # reward = self.default_policy(expand_node)
        ############################################
        if(expand_node.state.state.error is not None):
          reward = -1
        elif(expand_node.state.state.isFinish()):
          reward = 1
        else:
          encodestate = encode_state(expand_node.state.state.tacticState, self.args['feature_size'])
          input_value = torch.FloatTensor(np.array(encodestate).astype(np.float64)).to(self.device)
          reward = float(self.value_model(input_value))
        
        #############################################
        
        # 如果到达叶子节点(证明成功)，终止循环，开始寻找其所用到的所有策略

        # 3. Update all passing nodes with reward
        self.backup(expand_node, reward)
        

      # N. Get the best next node
      # best_next_node = self.best_child(node, False)
      # return best_next_node
      
      return node

    def runmcts(self, lean, time_out):
        node =  self.node
        computation_budget = 1000000000
        flag = False
        
        start = timeit.default_timer()
        
        # Run as much as possible under the computation budget
        for iteration in range(computation_budget):
          # print(iteration)
          if(iteration<215):
            # 1. Find the best node to expand
            expand_node = self.tree_policy(node, lean, False)
            
            if(expand_node.flag == -2):
              return flag
            # print("当前在第{}层, 父节点id为：{},".format(expand_node.depth,expand_node.parent.state.state.id))
            # if(expand_node.depth > 1):
            #   flag = 0
            #   for node_puct in expand_node.parent.parent.children: #父节点及其兄弟节点
            #     if(node_puct.state.state == 1 or node_puct.state.state == -1):
            #       if(flag == 0):
            #         print("当前节点的puct：{}".format(node_puct.puct))
            #         flag = 1
            #     else:
            #       print("当前节点的id为：{} puct：{}".format(node_puct.state.state.id, node_puct.puct))
            # print("父节点id为：{}".format(expand_node.parent.state.state.id))
            
            # print("节点状态为：{}".format(node.state.state))
            # print("当前选择策略为：{}".format(expand_node.state.tac))
            
            # # print("节点概率为：{}".format(expand_node.prob))   
            # if(expand_node.state.state.isFinish()):
            #   print("该策略有效")
            # elif(expand_node.state.state.error is not None):
            #   print("该策略失败")
            # else:
            #   # print("当前节点id为：{},证明未结束".format(expand_node.state.state.id))
            #   print("证明未结束")
              
            if(expand_node.state.state.isFinish()):
              flag = True
              print(expand_node)
              print("成功策略：")
              path = []
              state_list = []
              while expand_node.state.tac is not None:
                  path.append(expand_node.state.tac)  # 生成每步状态的策略
                  state_list.append(expand_node.state.state) # 每步的状态
                  expand_node = expand_node.parent
              path.reverse()
              state_list.append(node.state.state)
              state_list.reverse()
              
              print("成功路径策略：")
              for tatic in path:
                print(tatic)
              
              print("成功路径状态：")
              for state in state_list:
                print(state.tacticState)
                
              # end = timeit.default_timer()
              
              return flag

            # 2. Random run to add node and get reward
            # reward = self.default_policy(expand_node)
            ############################################
            
            if(expand_node.state.state.error is not None):
              reward = -1
            elif(expand_node.state.state.isFinish()):
              reward = 1
            else:
              encodestate = encode_state(expand_node.state.state.tacticState, self.args['feature_size'])
              input_value = torch.FloatTensor(np.array(encodestate).astype(np.float64)).to(self.device)
              reward = float(self.value_model(input_value))
              
            #############################################
            
            # 如果到达叶子节点(证明成功)，终止循环，开始寻找其所用到的所有策略

            # 3. Update all passing nodes with reward
            self.backup(expand_node, reward)
            
            end = timeit.default_timer()

            timespend = end - start 
            if(timespend > time_out):
              print("搜索超时")
              return flag
          else:
            # 1. Find the best node to expand
            expand_node = self.tree_policy(node, lean, False)
            
            if(expand_node.flag == -2):
              return flag
            # print("当前在第{}层, 父节点id为：{},".format(expand_node.depth,expand_node.parent.state.state.id))
            # if(expand_node.depth > 1):
            #   flag = 0
            #   for node_puct in expand_node.parent.parent.children: #父节点及其兄弟节点
            #     if(node_puct.state.state == 1 or node_puct.state.state == -1):
            #       if(flag == 0):
            #         print("当前节点的puct：{}".format(node_puct.puct))
            #         flag = 1
            #     else:
            #       print("当前节点的id为：{} puct：{}".format(node_puct.state.state.id, node_puct.puct))
            # print("父节点id为：{}".format(expand_node.parent.state.state.id))
            
            # print("节点状态为：{}".format(node.state.state))
            # print("当前选择策略为：{}".format(expand_node.state.tac))
            
            # # print("节点概率为：{}".format(expand_node.prob))   
            # if(expand_node.state.state.isFinish()):
            #   print("该策略有效")
            # elif(expand_node.state.state.error is not None):
            #   print("该策略失败")
            # else:
            #   # print("当前节点id为：{},证明未结束".format(expand_node.state.state.id))
            #   print("证明未结束")
              
            if(expand_node.state.state.isFinish()):
              flag = True
              print(expand_node)
              print("成功策略：")
              path = []
              state_list = []
              while expand_node.state.tac is not None:
                  path.append(expand_node.state.tac)  # 生成每步状态的策略
                  state_list.append(expand_node.state.state) # 每步的状态
                  expand_node = expand_node.parent
              path.reverse()
              state_list.append(node.state.state)
              state_list.reverse()
              
              print("成功路径策略：")
              for tatic in path:
                print(tatic)
              
              print("成功路径状态：")
              for state in state_list:
                print(state.tacticState)
                
              # end = timeit.default_timer()
              
              return flag

            # 2. Random run to add node and get reward
            # reward = self.default_policy(expand_node)
            ############################################
            
            if(expand_node.state.state.error is not None):
              reward = -1
            elif(expand_node.state.state.isFinish()):
              reward = 1
            else:
              encodestate = encode_state(expand_node.state.state.tacticState, self.args['feature_size'])
              input_value = torch.FloatTensor(np.array(encodestate).astype(np.float64)).to(self.device)
              reward = float(self.value_model(input_value))
              
            #############################################
            
            # 如果到达叶子节点(证明成功)，终止循环，开始寻找其所用到的所有策略

            # 3. Update all passing nodes with reward
            self.backup(expand_node, reward)
            
            end = timeit.default_timer()

            timespend = end - start 
            if(timespend > time_out):
              print("搜索超时")
              return flag
          # N. Get the best next node
          # best_next_node = self.best_child(node, False)
          # return best_next_node
        
        return 

def search(init_state, lean:Lean4Gym, max_iters=int(1e6), num_samples=16,
            max_tokens=255, time_out=600, queue_max_count=int(1e6)):
    queue = [(0.0, [], init_state)]
    visited = set() 
    proof_finished = False 
    start = time.time()

    for iteration in trange(max_iters):
        if len(queue) == 0 or proof_finished:
            break

        current_time = time.time()

        if current_time - start > time_out or len(queue) > queue_max_count:
            print("theorem is not proved")
            break
            

        total_score, steps, state = heapq.heappop(queue)
        # state.print()

        visited.add(state.getTacticState())

        step_cands, step_scores = generate_vllm(
            _prompt_proofstep(state.getTacticState()),
            model,
            tokenizer,
            [0],
            num_samples=num_samples,
            stop=tokenizer.eos_token,
            max_tokens=max_tokens
        )
        step_cands = [s.strip() for s in step_cands]

        # print(step_cands)
        # break

        for step, score in zip(step_cands, step_scores):
            # try:
                result = lean.run_tactic(state, [step])
                
                if result.getError() != None:
                    continue

                if result.isFinish():
                    proof_finished = True
                    
                    print("Theorem has proved!")
                    print(steps+[step])
                    current_time = time.time()
                    print("证明时长为：{}".format(current_time-start))
                    return proof_finished

                else:
                    if result.getTacticState() not in visited:
                        new_score = (total_score - score)
                        heapq.heappush(queue, (new_score, steps+[step], result)) 

            # except (Exception) as ex:
            #     print(ex)      
    return proof_finished
    
# def main():
#   # Create the initialized state and initialized node
#   init_state = State(state)
#   node = Node()
#   node.set_state(init_state)
#   current_node = node
#   mcts = MCTS(current_node)
#   current_node = mcts.run()
#   print("搜索完成")

# if __name__ == "__main__":
#   main()