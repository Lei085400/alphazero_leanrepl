# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# tokenizer = AutoTokenizer.from_pretrained("/home2/wanglei/python_project/leandojo-lean4-tacgen-byt5-small")       # Or "lean3" -> "lean4"
# model = AutoModelForSeq2SeqLM.from_pretrained("/home2/wanglei/python_project/leandojo-lean4-tacgen-byt5-small")

# state = "a b c : Nat\n⊢ a + (b + c) = a + c + b"
# tokenized_state = tokenizer(state, return_tensors="pt")

# # # Generate a single tactic.
# # tactic_ids = model.generate(tokenized_state.input_ids, max_length=1024)
# # tactic = tokenizer.decode(tactic_ids[0], skip_special_tokens=True)
# # print(tactic, end="\n\n")

# # Generate multiple tactics via beam search.
# tactic_candidates_ids = model.generate(
#     tokenized_state.input_ids,
#     max_length=1024,
#     num_beams=4,
#     length_penalty=0.0,
#     do_sample=False,
#     num_return_sequences=4,
#     early_stopping=False,
# )

# tactic_candidates = tokenizer.batch_decode(
#     tactic_candidates_ids, skip_special_tokens=True
# )
# for tac in tactic_candidates:
#     print(tac)
# print(tactic_candidates)
# print(type(tactic_candidates))
# # for tac in tactic_candidates:
# #     print(tokenizer.encode(tac))


import random
list1 = ['1', '2']
choose1 = random.choices([choice for choice in list1],k=1)
print(choose1)