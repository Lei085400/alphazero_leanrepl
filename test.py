import json
import heapq
import transformers
import subprocess
import vllm
import time
import timeit
from datetime import datetime
from tqdm import tqdm, trange
from pathlib import Path
from Lean4Gym import *
import copy
# from lean_dojo import *
import traceback

import torch
# os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu') 

model_name_or_path = "/home/wanglei/AAAI/lean_ATG/model/pythia2.8b_choose"

def list_files(directory):
    filelist = []
    for i, file in enumerate(os.listdir(directory)):
        if os.path.isfile(os.path.join(directory, file)):
            print(file)
            filelist.append(file)
    return filelist

count = 0
succ = []
succ_name = []

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
def generate_vllm(prompt, model, tokenizer, temperatures, num_samples, stop, max_tokens=256):
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

        # for tac in outputs[0].outputs:
        #     print(tac)

        if len(outputs) == 0:
            return [], []
        for output in outputs[0].outputs:
            text = output.text.replace(tokenizer.eos_token, '') 
            score = output.cumulative_logprob/max(len(output.token_ids), 1)
            texts.append(text)
            scores.append(score)

    texts, scores = _unique_sorted(texts, scores)
    return texts, scores


def search(file, init_state, lean:Lean4Gym, max_iters=int(1e6), num_samples=16,
            max_tokens=255, time_out=360, queue_max_count=int(1e6)):
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
            try:
                result = lean.run_tactic(state, [step])
                
                if result.getError() != None:
                    continue

                if result.isFinish():
                    proof_finished = True
                    
                    step_list = copy.copy(steps)
                    step_list.append(step)
                    # print(step_list)
                    # print(steps)
                    
                    FF = open(r'/home/wanglei/AAAI/lean_ATG/leanproject/alphazero_leanrepl/valid_succ.txt','a')
                    FF.write(file +'\n')
                    FF.close() 
                    
                    FF0 = open(r'/home/wanglei/AAAI/lean_ATG/leanproject/alphazero_leanrepl/valid_succ_step.txt','a')
                    FF0.write(str(step_list) + '\n')
                    FF0.close() 
                    
                    print("Theorem has proved!")
                    current_time = time.time()
                    print("证明时长为：{}".format(current_time-start))
                    return proof_finished

                else:
                    if result.getTacticState() not in visited:
                        new_score = (total_score - score)
                        heapq.heappush(queue, (new_score, steps+[step], result)) 

            except (Exception) as ex:
                print(ex)      
    return proof_finished
        

file_list = []
with open('/home/wanglei/AAAI/lean_ATG/leanproject/alphazero_leanrepl/example.txt', 'r') as file: 
    lines = file.readlines() 
    for line in lines:
        line = ''.join(line).strip('\n')
        file_list.append(line)
        print(line)

#待证明策略：
# lean_dir = "/home/wanglei/AAAI/lean_ATG/leanproject/testfolder/test"
# # lean_dir = "/home2/wanglei/Project/testfolder"
# file_list = list_files(lean_dir)
# # print(len(file_list))
# Fi = open(r'file_list_test.txt','w')
# for i in file_list:
#     Fi.write(str(i)+'\n')
# Fi.close() 


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
    # init_state = State(state)
    # node = Node()
    # node.set_state(init_state)


    start = timeit.default_timer()

    print("开始搜索策略")

    flag = search(file, state, lean)
    
    end = timeit.default_timer()
    
    if(flag == True):
        count += 1
        # succ.append(state.tacticState)
        succ_name.append(format(file))
        
    print ("所用时间：{}".format(str(end-start)))
    print("第{}个定理".format(str(i)))
    print("已成功证明{}条定理".format(str(count))) 
    F0 = open(r'/home/wanglei/AAAI/lean_ATG/leanproject/vaild_record.txt','a')
    F0.write("证明定理为:{}".format(file) +'\n')
    F0.write("所用时间：{}".format(str(end-start)) +'\n')
    F0.write("第{}个定理".format(str(i))+'\n')
    F0.write("已成功证明{}条定理".format(str(count))+'\n')
    F0.write("===================================="+'\n')
    F0.close() 
    
print("成功总数：{}".format(str(count)))
print("通过比例：{}".format(str(count/len(file_list))))
# print("成功定理有：")
# print(succ)
F = open(r'/home/wanglei/AAAI/lean_ATG/leanproject/alphazero_leanrepl/output_valid.txt','w')
for i in succ_name:
    F.write(str(i)+'\n')
F.close() 