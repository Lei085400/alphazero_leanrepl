import subprocess
import os
import time
import select
import json
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# 导入Lean4Gym
from Lean4Gym import Lean4Gym, ProofState

# tokenizer = AutoTokenizer.from_pretrained("/home2/wanglei/python_project/leandojo-lean4-tacgen-byt5-small")       # Or "lean3" -> "lean4"
# model = AutoModelForSeq2SeqLM.from_pretrained("/home2/wanglei/python_project/leandojo-lean4-tacgen-byt5-small")   # Or "lean3" -> "lean4"

# feature_size = 100
# def encode_state(state):
#     encode_state = tokenizer.encode(str(state))
#     if(len(encode_state)<=feature_size):
#         encode_state += [0]*(feature_size-len(encode_state))  #list
#     else:
#         encode_state = encode_state[:feature_size]
#     # print("encode")
#     # print(encode_state)
#     return encode_state

def main():

    lean_workdir = "/home2/wanglei/Project" # Lean工程的根目录
    lean_file = "demo.lean"   # 待证明定理的Lean文件
    print("lean_workdir:", lean_workdir)
    print("lean_file   :", lean_file)


    lean = Lean4Gym(lean_workdir, lean_file)

    stat0 = lean.getInitState()
    # print("stat0:")
    # print(stat0.tacticState)
    # print(encode_state(stat0.tacticState))
    
    stat1 = lean.run_tactic(stat0, ["have choose_eq_choose_sub_add :  choose n k = choose (n - 1 + 1) (k - 1 + 1)  :=  by  rw[Nat.sub_add_cancel h1, Nat.sub_add_cancel h2]"])
    state = stat1.getTacticState()
    # print(")))))))))))))))))))))))))))))")
    print(state)
    # print(")))))))))))))))))))))))))))))")
    # stat1.print()
    # stat1 = lean.run_tactic(stat0, ["sorry"])
    # stat1.print()
    # stat1 = lean.run_tactic(stat0, ["rw[Nat.sub_add_cancel h1, Nat.sub_add_cancel h2]"])
    # stat1.print()
    stat2 = lean.run_tactic(stat1, [" rw[choose_eq_choose_sub_add]"])
    # stat2.print()
    # stat3 = lean.run_tactic(stat1, ["rw[choose_eq_choose_sub_add]"])
    # stat3.print()
    stat3 = lean.run_tactic(stat1, ["rw[add_comm (choose (n - 1) k) (choose (n - 1) (k - 1))] "])
    stat3.print()
    stat4 = lean.run_tactic(stat1, ["have choose_sub_eq_choose_sub_add : choose (n - 1) k = choose (n - 1) (k - 1 + 1) := by rw[Nat.sub_add_cancel h2] "])
    stat4.print()
    print("伊万·费尧多罗维奇·卡拉马佐夫")
    # stat5 = lean.run_tactic(stat4, ["rw[choose_sub_eq_choose_sub_add, choose_succ_succ] "])
    stat5 = lean.run_tactic(stat1, ["rw[add_assoc] "])
    stat5.print()
    
    stat5 = lean.run_tactic(stat4, ["rw[choose_sub_eq_choose_sub_add, choose_succ_succ] "])
    stat5.print()
    
    print(stat5.isFinish())

    lean.close()


if __name__ == "__main__":
    main()
