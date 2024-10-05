import json
import numpy as np
from src.gritlm import GritLM
from scipy.spatial.distance import cosine
from tqdm import tqdm

KEY_WORDS = ['endmodule', 'end', 'endcase', 'else', 'begin']
INSTRUCT_LOC = 'Now you are a verilog designer. You are given the design description and buggy verilog code segment. Find the buggy line in the code segment.\n'
INSTRUCT_GEN = 'Now you are a verilog designer. You need to fix the bug in the buggy code segment:\n'
INSTRUCT_CLS = 'Now you are a verilog designer. You are given the design description and buggy verilog code segment. Infer the bug type in the code segment.\n'
INSTRUCT_REP = 'Represent this text: '
INSTRUCT_DESC = 'The bug type is '
JSON_FORMAT = '{"buggy_code": "The buggy code in the systemverilog (just one line of code)", "correct_code": "The correct code (just one line of code that can directly replace the buggy code, without any other description)"}'
BUG_TYPE = {
    'width': 0, 'logic': 0, 'assignment': 0, 'initial': 0, 'data': 0,
    'state': 0, 'others': 0, 'comparison': 0, 'bitwise': 0, 'condition': 0,
    'signal': 0, 'arithmetic': 0, 'value': 0
}
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


def list2str(ll):
    s = '[\n'
    for i in ll:
        s += (i + '\n')
    s += ']'
    return s


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


dataset_gen = []
with open("data/test/generative.jsonl", "r") as json_file:
    json_list = list(json_file)
    for json_str in json_list:
        dataset_gen.append(json.loads(json_str))

# Loads the model for both capabilities; If you only need embedding pass `mode="embedding"` to save memory (no lm head)
model_path = "./VeriDebug"
model = GritLM(model_path, torch_dtype="auto", device_map="auto")
print(f"Model loaded from {model_path}")

# Embedding/Representation ###
NUM_INSTANCES = -1
if NUM_INSTANCES == -1:
    NUM_INSTANCES = len(dataset_gen)
cosine_sim_dataset = []
poses = []
documents = []
gt_dataset = []
pred_dataset = []
with tqdm(total=NUM_INSTANCES) as pbar:
    for data in dataset_gen[:NUM_INSTANCES]:
        pbar.update(1)

        module_spec = data['spec']
        buggy_code = data['buggy_code']
        original = data['original']
        gt_dataset.append(original.strip().strip('\r').strip('\t'))

        input = f'{module_spec}\n\nBuggy code: \n{buggy_code}'
        query_loc = f'{INSTRUCT_LOC}{input}'
        _, buggy_code_lines = gen_neg(buggy_code)

        # rank buggy lines by cosine similarity
        query_loc = [query_loc]
        q_rep = model.encode(
            query_loc, instruction=gritlm_instruction(INSTRUCT_REP), max_length=4096)
        d_1 = buggy_code_lines[:len(buggy_code_lines) // 2]
        d_2 = buggy_code_lines[len(buggy_code_lines) // 2:]
        d_rep_1 = model.encode(
            d_1, instruction=gritlm_instruction(""), max_length=128)
        d_rep_2 = model.encode(
            d_2, instruction=gritlm_instruction(""), max_length=128)
        d_rep = np.concatenate([d_rep_1, d_rep_2], axis=0)
        # select top 5
        cosine_sim = [1 - cosine(q_rep[0], d) for d in d_rep]
        sim_rank = np.argsort(cosine_sim)[::-1]
        buggy_code_lines_ranked = [buggy_code_lines[i] for i in sim_rank][:5]
        buggy_code_lines_ranked = list2str(buggy_code_lines_ranked)

        # rank buugy types by cosine similarity
        query_cls = f'{INSTRUCT_CLS}{input}'
        d_types = [INSTRUCT_DESC + b + '.' + BUG_DESC[b]
                   for b in BUG_TYPE.keys()]
        q_rep = model.encode(
            query_loc, instruction=gritlm_instruction(INSTRUCT_REP), max_length=4096)
        d_rep = model.encode(
            d_types, instruction=gritlm_instruction(""), max_length=128)
        # select top 5
        cosine_sim = [1 - cosine(q_rep[0], d) for d in d_rep]
        sim_rank = np.argsort(cosine_sim)[::-1]
        buggy_type_ranked = [d_types[i] for i in sim_rank][:3]
        buggy_type_ranked = list2str(buggy_type_ranked)

        # fix the bug
        input = f'{INSTRUCT_GEN}{buggy_code}\n\nThe specification file of this code is:\n{module_spec}\n\nThe possible buggy lines ranking list are:\n{buggy_code_lines_ranked}\n\nThe possible bug type ranking list are:\n{buggy_type_ranked}\n\nYour task is to return me a json to analyze how the code should be modified, in the following format:\n{JSON_FORMAT}.'

        messages = [
            {"role": "user", "content": input},
        ]
        encoded = model.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt")
        encoded = encoded.to(model.device)
        gen = model.generate(encoded, max_new_tokens=256, do_sample=True)
        valid_gen = gen[:, encoded.shape[1]:]
        decoded = model.tokenizer.batch_decode(valid_gen)
        # truncate the decoded text before </s>
        decoded = [d[:d.find('}')+1] for d in decoded]
        try:
            decoded_dict = json.loads(decoded[0])
            pred_dataset.append(
                decoded_dict['correct_code'].strip().strip('\r').strip('\t'))
            print(decoded_dict['correct_code'])
        except Exception as e:
            # This catches any other unexpected exceptions
            print(f"An unexpected error occurred: {e}")
            pred_dataset.append('')
            print(decoded[0])
        print("===")
        print(original)
        print("------------------------")


acc = []
for candidate, gt in zip(pred_dataset, gt_dataset):
    if candidate == gt:
        acc.append(1)
acc = len(acc) / NUM_INSTANCES

print(f"Accuracy@1: {acc};")
