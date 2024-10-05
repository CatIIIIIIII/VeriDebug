import json
import numpy as np
from src.gritlm import GritLM
from scipy.spatial.distance import cosine
from tqdm import tqdm

KEY_WORDS = ['endmodule', 'end', 'endcase', 'else', 'begin']


def gen_neg(buggy_code):

    buggy_code_lines = buggy_code.split('\n')
    buggy_code_lines = [line.strip() for line in buggy_code_lines]
    buggy_code_lines = [line.strip('\t') for line in buggy_code_lines]
    buggy_code_lines = [line.strip('\r') for line in buggy_code_lines]
    buggy_code_lines = [line for line in buggy_code_lines if len(line) > 0]
    # remove comments
    buggy_code_lines_neg = [
        line for line in buggy_code_lines if not line.startswith('//') and not line.startswith('*') and not line.startswith('/*') and line not in KEY_WORDS]
    # remove not useful lines
    buggy_code_lines_neg = [
        line for line in buggy_code_lines_neg if ' ' in line and len(line.replace(' ', '')) > 4]

    return buggy_code_lines, buggy_code_lines_neg


def gritlm_instruction(instruction):
    return "<|user|>\n" + instruction + "\n<|embed|>\n" if instruction else "<|embed|>\n"


dataset_embed = []
with open("data/test/embedding.jsonl", "r") as json_file:
    json_list = list(json_file)
    for json_str in json_list:
        dataset_embed.append(json.loads(json_str))

# Loads the model for both capabilities; If you only need embedding pass `mode="embedding"` to save memory (no lm head)
model_path = "./VeriSeek"
model = GritLM(model_path, torch_dtype="auto", mode="embedding")
print(f"Model loaded from {model_path}")

### Embedding/Representation ###
NUM_INSTANCES = -1
if NUM_INSTANCES == -1:
    NUM_INSTANCES = len(dataset_embed)
cosine_sim_dataset = []
poses = []
documents = []
buggy_candidates_dataset = []
buggy_gt_dataset = []
with tqdm(total=NUM_INSTANCES) as pbar:
    for data in dataset_embed[:NUM_INSTANCES]:
        pbar.update(1)

        query, buggy_gt, buggy_code = data['line_query'][1], data['line'], data['buggy_code']
        _, buggy_code_lines = gen_neg(buggy_code)

        query = [query]
        # No need to add instruction for retrieval documents
        q_rep = model.encode(
            query, instruction=gritlm_instruction("Represent this text:"), max_length=4096)
        d_1 = buggy_code_lines[:len(buggy_code_lines) // 2]
        d_2 = buggy_code_lines[len(buggy_code_lines) // 2:]
        d_rep_1 = model.encode(
            d_1, instruction=gritlm_instruction(""), max_length=128)
        d_rep_2 = model.encode(
            d_2, instruction=gritlm_instruction(""), max_length=128)
        d_rep = np.concatenate([d_rep_1, d_rep_2], axis=0)

        cosine_sim = [1 - cosine(q_rep[0], d) for d in d_rep]
        sim_rank = np.argsort(cosine_sim)[::-1]
        buggy_code_lines_ranked = [buggy_code_lines[i] for i in sim_rank]
        buggy_candidates_dataset.append(buggy_code_lines_ranked[:5])
        buggy_gt_dataset.append(buggy_gt)

acc, acc_3, acc_5 = [], [], []
for candidate, gt in zip(buggy_candidates_dataset, buggy_gt_dataset):
    gt = gt[0]
    if candidate[0] == gt:
        acc.append(1)
    if gt in candidate[:3]:
        acc_3.append(1)
    if gt in candidate[:5]:
        acc_5.append(1)
    print(candidate, gt)
    print("===")
acc = len(acc) / NUM_INSTANCES
acc_3 = len(acc_3) / NUM_INSTANCES
acc_5 = len(acc_5) / NUM_INSTANCES

print(f"Accuracy@1: {acc}; Accuracy@3: {acc_3}; Accuracy@5: {acc_5}")

