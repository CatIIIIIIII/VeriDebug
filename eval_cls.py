import re
import json
import numpy as np
from src.gritlm import GritLM
from scipy.spatial.distance import cosine
from tqdm import tqdm

INSTRUCT_DESC = f'The bug type is '
BUG_TYPE = ['width', 'logic', 'assignment', 'initial', 'data', 'state',
            'others', 'comparison', 'bitwise', 'condition', 'signal', 'arithmetic', 'value']
BUG_DESC = {
    'width': 'Mismatched bit widths in assignments, operations, or port connections, leading to unintended truncation or zero-extension.',
    'logic': 'Errors in combinational or sequential logic design, resulting in incorrect circuit behavior or timing issues.',
    'assignment': 'Improper use of blocking (=) or non-blocking (<=) assignments, causing race conditions or unexpected signal updates.',
    'initial': 'Incorrect initialization of variables or registers, leading to undefined behavior or simulation mismatches.',
    'data': 'Errors in data handling, such as incorrect data types, improper conversions, or misuse of signed/unsigned values.',
    'state': 'Flaws in finite state machine (FSM) design, including missing states, incorrect transitions, or improper state encoding.',
    'others': 'Miscellaneous errors that don\'t fit into other categories, such as syntax errors or tool-specific issues.',
    'comparison': 'Incorrect use of equality (==) or inequality (!=) operators, or misuse of case equality (===) and case inequality (!==).',
    'bitwise': 'Errors in bitwise operations, including incorrect use of AND, OR, XOR, or shift operators.',
    'condition': 'Flaws in conditional statements (if-else, case) leading to incorrect branching or priority encoding issues.',
    'signal': 'Errors related to signal declarations, including incorrect use of wire/reg, input/output ports, or signal naming conflicts.',
    'arithmetic': 'Mistakes in arithmetic operations, such as overflow/underflow issues or incorrect use of signed/unsigned arithmetic.',
    'value': 'Incorrect constant values, parameter definitions, or literal representations leading to unexpected circuit behavior.'
}


def gritlm_instruction(instruction):
    return "<|user|>\n" + instruction + "\n<|embed|>\n" if instruction else "<|embed|>\n"

def extract_bug_types(text):
    # Define the regex pattern
    pattern = r'The bug type is (\w+)'
    
    # Find all matches
    matches = re.findall(pattern, text)
    
    return matches


dataset_embed = []
with open("data/test/embedding.jsonl", "r") as json_file:
    json_list = list(json_file)
    for json_str in json_list:
        dataset_embed.append(json.loads(json_str))

model_path = "./VeriDebug"
model = GritLM(model_path, torch_dtype="auto", mode="embedding")
print(f"Model loaded from {model_path}")

### Embedding/Representation ###
NUM_INSTANCES = -1
if NUM_INSTANCES == -1:
    NUM_INSTANCES = len(dataset_embed)
buggy_candidates_dataset = []
buggy_gt_dataset = []
with tqdm(total=NUM_INSTANCES) as pbar:
    for data in dataset_embed[:NUM_INSTANCES]:
        pbar.update(1)

        query, type_gt, buggy_code = data['type_query'][1], data['type'], data['buggy_code']

        query = [query]
        d = [INSTRUCT_DESC + t + '.' + BUG_DESC[t] for t in BUG_TYPE]
        # No need to add instruction for retrieval documents
        q_rep = model.encode(
            query, instruction=gritlm_instruction("Represent this text:"), max_length=4096)
        d_rep = model.encode(d, instruction=gritlm_instruction(""), max_length=128)

        cosine_sim = [1 - cosine(q_rep[0], d) for d in d_rep]
        sim_rank = np.argsort(cosine_sim)[::-1]
        buggy_code_types_ranked = [extract_bug_types(d[i])[0] for i in sim_rank]
        buggy_candidates_dataset.append(buggy_code_types_ranked[:3])
        buggy_gt_dataset.append(type_gt)

acc, acc_3 = [], []
for candidate, gt in zip(buggy_candidates_dataset, buggy_gt_dataset):
    gt = gt[0]
    if candidate[0] == gt:
        acc.append(1)
    if gt in candidate[:3]:
        acc_3.append(1)
    print(candidate, gt)
    print("===")
acc = len(acc) / NUM_INSTANCES
acc_3 = len(acc_3) / NUM_INSTANCES

print(f"Accuracy@1: {acc}; Accuracy@3: {acc_3}")

