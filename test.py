import re
import json
import numpy as np
from src.gritlm import GritLM
from scipy.spatial.distance import cosine

KEY_WORDS = ['endmodule', 'end', 'endcase', 'else', 'begin']
REP_QUERY = 'Represent this text: '
LINE_QUERY = """Now you are a verilog designer. You are given the design description and buggy verilog code segment. Infer the bug type in the code segment."""
TYPE_QUERY = "Now you are a verilog designer. You are given the design description and buggy verilog code segment. Infer the bug type in the code segment."
CLS_QUERY = 'Now you are a verilog designer. You are given the design description and buggy verilog code segment. Infer the bug type in the code segment.\n'
CLS_DESC = 'The bug type is '
BUG_CLS = {
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
GEN_INST = 'Now you are a verilog designer. You need to fix the bug in the buggy code segment:\n'
SPEC = """
---
### Module Specification: Cfu
# #### 1. Overview
# The `Cfu` (Custom Function Unit) module is designed to perform a simple selection operation based on the input command. It processes two 32-bit inputs and outputs one of them based on the least significant bit of the function ID. The module operates synchronously with a clock signal and uses a simple handshake protocol for command acceptance and response delivery.
# #### 2. Interface Description
# ##### Inputs:
# - **cmd_valid** (`input`): A signal indicating if the command inputs are valid.
# - **cmd_payload_function_id** (`input [9:0]`): A 10-bit function identifier which determines the operation of the module. Currently, only the LSB is used for selecting the output.
# - **cmd_payload_inputs_0** (`input [31:0]`): A 32-bit input data.
# - **cmd_payload_inputs_1** (`input [31:0]`): Another 32-bit input data.
# - **rsp_ready** (`input`): A signal from the downstream component indicating it is ready to accept the response.
# - **reset** (`input`): Asynchronous reset signal.
# - **clk** (`input`): Clock signal.
# ##### Outputs:
# - **cmd_ready** (`output`): A signal indicating the module is ready to accept a command.
- **rsp_valid** (`output`): A signal indicating that the response is valid and ready to be read.
# - **rsp_payload_outputs_0** (`output [31:0]`): The 32-bit output data, which is one of the two input data values based on the function ID.
#### 3. Functional Description
##### Command and Response Protocol:
- **Handshake Mechanism**: The module uses a simple handshake mechanism for command acceptance and response delivery. The `cmd_ready` signal is asserted when the module is ready to accept a new command, which depends on the `rsp_ready` signal. The `rsp_valid` signal is asserted when the module has a valid response ready, which is directly tied to the `cmd_valid` signal.
  
##### Data Processing:
- **Output Selection**: The output, `rsp_payload_outputs_0`, is selected based on the least significant bit (LSB) of `cmd_payload_function_id`. If the LSB is 0, `rsp_payload_outputs_0` is equal to `cmd_payload_inputs_0`. If the LSB is 1, `rsp_payload_outputs_0` is equal to `cmd_payload_inputs_1`.
#### 4. Timing and Synchronization
- The module operates synchronously with respect to the provided clock signal (`clk`). All inputs are sampled, and outputs are updated on the rising edge of the clock.
- The reset (`reset`) is asynchronous and active-high, which means all internal states and outputs are reset when `reset` is asserted, regardless of the clock.
#### 5. Use Cases
- **Simple Data Selector**: This module can be used in systems where conditional data forwarding is needed based on a simple configuration or status bit.
# - **Function ID Expansion**: While currently only the LSB of the function ID is used, the module can be expanded to use more bits for more complex selection logic or operations.
#### 6. Limitations and Future Enhancements
# - **Function ID Utilization**: Currently, only the LSB of the function ID is used. Future enhancements could include decoding more bits to perform different operations.
- **Pipeline Stages**: The module is purely combinational regarding the data path. Adding pipeline stages could help in meeting timing requirements for higher clock frequencies.
---
This specification provides a detailed overview of the `Cfu` module's functionality, interface, and behavior based on the provided Verilog code. It outlines the basic operation, use cases, and potential areas for future enhancements.

Buggy code: 

"""
BUGGY = """
module Cfu (
    input               cmd_valid,
    output              cmd_ready,
    input      [9:0]    cmd_payload_function_id,
    input      [31:0]   cmd_payload_inputs_0,
    input      [31:0]   cmd_payload_inputs_1,
    output              rsp_valid,
    input               rsp_ready,
    output     [31:0]   rsp_payload_outputs_0,
    input               reset,
    input               clk
    );
    // Trivial handshaking for a combinational CFU
    assign rsp_valid = cmd_valid;
    assign cmd_ready = rsp_ready | cmd_valid;
    //
    // select output -- note that we're not fully decoding the 3 function_id bits
    //
    assign rsp_payload_outputs_0 = cmd_payload_function_id[0] ? 
    cmd_payload_inputs_1 :
    cmd_payload_inputs_0 ;
    endmodule
"""
JSON_FORMAT = '{"buggy_code": "The buggy code in the systemverilog (just one line of code)", "correct_code": "The correct code (just one line of code that can directly replace the buggy code, without any other description)"}'
BUGGY_LINE_GT = "assign cmd_ready = rsp_ready | cmd_valid;"
BUGGY_CLS_GT = "logic"
FIX_GT = "assign cmd_ready = rsp_ready;"


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


def extract_bug_types(text):
    # Define the regex pattern
    pattern = r'The bug type is (\w+)'
    
    # Find all matches
    matches = re.findall(pattern, text)
    
    return matches

# Loads the model for both capabilities; If you only need embedding pass `mode="embedding"` to save memory (no lm head)
model_path = "./VeriDebug"
model = GritLM(model_path, torch_dtype="auto", mode="unified")
print(f"Model loaded from {model_path}")

### Embedding/Representation ###
# buggy line location
_, buggy_code_lines = gen_neg(BUGGY)
query = [LINE_QUERY + "\n" + SPEC + "\n" + BUGGY]
q_rep = model.encode(query, 
                     instruction=gritlm_instruction("Represent this text:"), max_length=4096)
d_rep = model.encode(buggy_code_lines, 
                     instruction=gritlm_instruction(""), 
                     max_length=128)
cosine_sim = [1 - cosine(q_rep[0], d) for d in d_rep]
sim_rank = np.argsort(cosine_sim)[::-1]
buggy_code_lines_ranked = [buggy_code_lines[i] for i in sim_rank]
print("========== Buggy code lines ranked by similarity ==========")
print(f"Buggy code lines candidates (ranked by similarity): \n{buggy_code_lines_ranked}")
print("----------------------------------------")
print(f"Ground truth: {BUGGY_LINE_GT}")
print("===========================================================")

# buggy type classification
query = [TYPE_QUERY + "\n" + SPEC + "\n" + BUGGY]
d_types = [CLS_DESC + b + '.' + BUG_DESC[b]
                   for b in BUG_CLS.keys()]
q_rep = model.encode(query, 
                     instruction=gritlm_instruction(REP_QUERY), 
                     max_length=4096)
d_rep = model.encode(d_types, 
                     instruction=gritlm_instruction(""), 
                     max_length=128)
cosine_sim = [1 - cosine(q_rep[0], d) for d in d_rep]
sim_rank = np.argsort(cosine_sim)[::-1]
buggy_type_ranked = [d_types[i] for i in sim_rank]
print("============ Buggy type ranked by similarity ==============")
print(f"Buggy type candidates (ranked by similarity): \n{[extract_bug_types(i)[0] for i in buggy_type_ranked]}")
print("----------------------------------------")
print(f"Ground truth: {BUGGY_CLS_GT}")
print("===========================================================")


### Generation ###
instruct = f'{GEN_INST}{BUGGY}\n\nThe specification file of this code is:\n{SPEC}\n\nThe possible buggy lines ranking list are:\n{buggy_code_lines_ranked}\n\nThe possible bug type ranking list are:\n{buggy_type_ranked}\n\nYour task is to return me a json to analyze how the code should be modified, in the following format:\n{JSON_FORMAT}.'
messages = [
            {"role": "user", "content": instruct},
        ]
encoded = model.tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt")
encoded = encoded.to(model.device)
gen = model.generate(encoded, max_new_tokens=256, do_sample=True)
valid_gen = gen[:, encoded.shape[1]:]
decoded = model.tokenizer.batch_decode(valid_gen)
# truncate the decoded text before </s>
decoded = [d[:d.find('}')+1] for d in decoded]
decoded_dict = json.loads(decoded[0])
print("==================== Buggy fix ============================")
print(f"Fix result: {decoded_dict}")
print("----------------------------------------")
print(f"Ground truth: {FIX_GT}")
print("===========================================================")
