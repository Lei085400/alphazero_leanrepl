import os
import numpy as np
from random import shuffle
import random
import copy
import torch
import torch.optim as optim
import pickle
from mcts import MCTS
from mcts import State
from mcts import Node
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM

# feature_size = 100

# tokenizer = AutoTokenizer.from_pretrained("/home2/wanglei/python_project/leandojo-lean4-tacgen-byt5-small")       # Or "lean3" -> "lean4"
# model = AutoModelForSeq2SeqLM.from_pretrained("/home2/wanglei/python_project/leandojo-lean4-tacgen-byt5-small")   # Or "lean3" -> "lean4"

tokenizer = AutoTokenizer.from_pretrained("/home/wanglei/AAAI/lean_ATG/leanproject/leandojo-lean4-tacgen-byt5-small")       # Or "lean3" -> "lean4"
model = AutoModelForSeq2SeqLM.from_pretrained("/home/wanglei/AAAI/lean_ATG/leanproject/leandojo-lean4-tacgen-byt5-small")   # Or "lean3" -> "lean4"


def encode_state(state, feature_size):
    encode_state = tokenizer.encode(str(state))
    if(len(encode_state)<=feature_size):
        encode_state += [0]*(feature_size-len(encode_state))  #list
    else:
        encode_state = encode_state[:feature_size]
    # print("encode")
    # print(encode_state)
    return encode_state

def encode_tactic(tactic,feature_size):
    encode_tactic = tokenizer.encode(str(tactic))
    if(len(encode_tactic)<=feature_size):
        encode_tactic += [0]*(feature_size-len(encode_tactic))
    else:
        encode_tactic = encode_tactic[:feature_size]
    return encode_tactic
    
class Trainer:

    def __init__(self, policy_model, value_model, args, device):
        self.policy_model = policy_model
        self.value_model = value_model
        self.args = args
        self.mcts = None
        self.device = device
        # self.mcts = MCTS(node, self.policy_model, self.value_model, self.args)


    def exceute_episode(self, root): # 返回当前搜索树大步路径上 所有 点的 概率、奖励、状态、策略 作为训练样本
        # print("开始采样")
        policy_examples = []
        value_examples = []
        # state = self.game.get_init_board()
        node = root
        state = node.state

        while True:  #循环一次是往下走一个节点，探索大步节点路径，直到证明结束（成功或失败），每一次选择孩子节点中价值最高的策略，并计算其选择概率，再将其状态和策略全部放入样本列表
            # canonical_board = self.game.get_canonical_board(state, current_player)
            reward = 0
            reward0 = 0
            max_times = 0
            max_i = 0
            finish = 0
            action_probs = [0 for _ in range(self.args['TACRIC_NUMBER'])]
            for index, children_node in enumerate(node.children):
                action_probs[index] = children_node.visit_times  
                if(action_probs[index] > max_times): # 找到当前节点中概率最大(访问次数最多)的子节点
                    max_times = action_probs[index]
                    max_i = index
            # print("节点")
            # print(action_probs)
            if(np.sum(action_probs)!=0):
                action_probs = action_probs / np.sum(action_probs)  #计算每个节点的概率值，当前节点node暂时没有子节点时，即[0,0,0,0]时，会报错

            encodestate = encode_state(state.state.tacticState, self.args['feature_size'])
            for index, children_node in enumerate(node.children):  #记录每个大步节点node的所有策略的概率，放入train_examples
                encodetactic = encode_tactic(children_node.state.tac, self.args['feature_size'])
                input_policy = encodestate + encodetactic
                policy_examples.append((input_policy, action_probs[index]))
                
                children_state = children_node.state
                
                children_encodestate = encode_state(children_state.state.tacticState, self.args['feature_size'])
                if(children_state.state.error is not None):
                    print(children_encodestate)
                
                reward = children_state.compute_reward()  #当前节点的价值
                reward0 = reward
                if(reward is None):
                    reward0 = 0
                value_examples.append((children_encodestate, reward0)) #state为已完成或者失败时，reward直接为1or0
            
            try: #向下走一个节点，选择概率最大的策略执行
                node = node.children[max_i]
                state = node.state
                # encodestate = encode_state(state.state)
                # reward = state.compute_reward()  #当前节点的价值
                # reward0 = reward
                # if(reward is None):
                #     reward0 = 0
                # value_examples.append((encodestate, reward0)) #state为已完成或者失败时，reward直接为1or0
            except:
                finish = 1
                
            if (reward is not None or finish == 1): 
                policy = []
                value = []
                for hist_input, hist_action_probs in policy_examples:
                    policy.append((hist_input, hist_action_probs))

                for hist_state, hist_reward in value_examples:
                    value.append((hist_state, hist_reward))

                # print("状态样本为：")
                # print(hist_state)
                # print("概率样本为：")
                # print(hist_action_probs)
                return policy, value


    def learn(self, state_list, lean_list): 
        print("训练开始啦")
        policy_train_example = []
        value_train_example = []
        for index, state in enumerate(state_list):
            for i in range(1, self.args['numIters'] + 1):  # 每次迭代训练一个证明目标（即一个根节点node）
                print("{}/{}".format(i, self.args['numIters']))
                
                count = 0
                # node_ = copy.copy(node)
                node = Node()
                node.set_state(state)
                for j in range (self.args['numEps']): #对该证明目标训练的循环迭代次数，迭代一次，生成一棵搜索树，当迭代次数% b = 0， 则采样当前搜索树的大步节点，执行下列步骤 
                    # print("第{}轮".format(j))
                    self.mcts = MCTS(node, self.policy_model, self.value_model, self.args, self.device)
                    # root = self.mcts.run()
                    # print(node.state)
                    node = self.mcts.run(lean_list[index])
                    count += 1

                    if(count % 5 == 0):
                    # 如果迭代次数 % b = 0， 则采样，执行下列步骤 
                        # print("采样一次")
                        policy_train_examples, value_train_examples = self.exceute_episode(node)  # 都是列表，返回大步节点所构成的一条路径上所有样本数据。采样所有大步节点路径上的节点，每次循环返回一个训练样本，即一对（状态、策略）对 和 一个状态，及其相应的概率和价值
                        policy_train_example.extend(policy_train_examples)
                        value_train_example.extend(value_train_examples)
        
        shuffle(policy_train_example)
        shuffle(value_train_example)
        with open('policy_train_example.pickle', 'wb') as f:
            pickle.dump(policy_train_example, f)
        with open('value_train_example.pickle', 'wb') as f:
            pickle.dump(value_train_example, f)
        self.train()
        return
            
    def train(self):
        print("xunlian")
        with open('policy_train_example.pickle', 'rb') as f:
            policy_train_example = pickle.load(f)
        with open('value_train_example.pickle', 'rb') as f:
            value_train_example = pickle.load(f)
        self.policy_train(policy_train_example)  # 每次训练样本：当前mcts树中，numEps/10条大步节点路径上所有节点
        self.value_train(value_train_example)
        # filename = self.args['checkpoint_path']
        self.save_checkpoint_policy(folder=".", filename="policy_model")  
        self.save_checkpoint_value(folder=".", filename="value_model") 
        return

    def policy_train(self, policy_examples):
        
        # print("开始策略概率的训练")
        optimizer = optim.Adam(self.policy_model.parameters(), lr=5e-4)
        pi_losses = []
        num_batches = len(policy_examples) // self.args['batch_size']
        
        for epoch in range(self.args['epochs']):
            self.policy_model.train()
            
            shuffle(policy_examples)
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * self.args['batch_size']
                end_idx = (batch_idx + 1) * self.args['batch_size']
            # while batch_idx < int(len(policy_examples) / self.args['batch_size']):
                batch_examples = policy_examples[start_idx:end_idx]
                input, target = list(zip(*[(example[0], example[1]) for example in batch_examples]))
                input = torch.FloatTensor(np.array(input).astype(np.float64)).to(self.device)
                target = torch.FloatTensor(np.array(target).astype(np.float64)).to(self.device)

                # predict
                input = input.contiguous()
                target_pis = target.contiguous()
                
                
                out_pi = self.policy_model(input)
                l_pi = self.loss_pi(target_pis, out_pi)

                pi_losses.append(float(l_pi))

                optimizer.zero_grad()
                l_pi.backward()
                optimizer.step()

               
            # print(pi_losses)
            # print("Policy Loss", np.mean(pi_losses))
            
            # print("Examples:")
            # print(outpi[0].detach())
            # print(targetpis[0])


    def value_train(self, examples):
        # print("开始value的训练")
        optimizer = optim.Adam(self.value_model.parameters(), lr=5e-4)
        v_losses = []
        num_batches = len(examples) // self.args['batch_size']
        
        for epoch in range(self.args['epochs']):
            self.value_model.train()

            random.shuffle(examples)
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * self.args['batch_size']
                end_idx = (batch_idx + 1) * self.args['batch_size']
            # while batch_idx < int(len(policy_examples) / self.args['batch_size']):
                batch_examples = examples[start_idx:end_idx]
                input, target = list(zip(*[(example[0], example[1]) for example in batch_examples]))
        
                input = torch.FloatTensor(np.array(input).astype(np.float64)).to(self.device)
                target = torch.FloatTensor(np.array(target).astype(np.float64)).to(self.device)

                # predict
                boards = input.contiguous()
                target_vs = target.contiguous()
                
                # compute output
                out_v = self.value_model(boards)
                l_v = self.loss_pi(target_vs, out_v)

                v_losses.append(float(l_v))

                optimizer.zero_grad()
                l_v.backward()
                optimizer.step()

            # print(v_losses)
            # print("Value Loss", np.mean(v_losses))
            # print("Examples:")
            # print(out_pi[0].detach())
            # print(target_pis[0])

    # def loss_pi(self, targets, outputs):
    #     loss = -(targets * torch.log(outputs)).sum(dim=1)
    #     return loss.mean()
    
    def loss_pi(self,targets, outputs):
        loss = torch.sum((targets - outputs) ** 2) / targets.size()[0]
        return loss

    def loss_v(self, targets, outputs):
        loss = torch.sum((targets-outputs.view(-1))**2)/targets.size()[0]
        return loss

    def save_checkpoint_policy(self, folder, filename):
        if not os.path.exists(folder):
            os.mkdir(folder)

        filepath = os.path.join(folder, filename)
        torch.save({
            'state_dict': self.policy_model.state_dict(),
        }, filepath)
        
        
    def save_checkpoint_value(self, folder, filename):
        if not os.path.exists(folder):
            os.mkdir(folder)

        filepath = os.path.join(folder, filename)
        torch.save({
            'state_dict': self.value_model.state_dict(),
        }, filepath)

