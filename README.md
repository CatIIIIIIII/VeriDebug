# VeriDebug
This is the experimental code and all materials for the paper "VeriDebug: A Unified LLM for Verilog Debugging via Contrastive Embedding and Guided Correction". The training dataset is located at [Hugging Face](https://huggingface.co/datasets/WANGNingroci/BuggyVerilog/). The model checkpoint is hosted on [Hugging Face](https://huggingface.co/WANGNingroci/VeriDebug/). 

## Table of Contents
- [Update Log](#Update)
- [Installation](#installation)
- [Training](#Training)
- [Inference](#Inference)
- [Citation](#Citation)

## Update Log
- 2024-10-04: Initial release of the code and materials.

## Installation
Use the following command to clone the repository:
```bash
git clone https://github.com/CatIIIIIIII/VeriDebug.git
cd VeriDebug
```
### Pre-requisites
install the required torch-related packages:
```bash
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```
install GradCache for the gradient caching:
```bash
git clone https://github.com/luyug/GradCache
cd GradCache
pip install .
cd ..
```

## Training
### Base Model Download
Download the pre-trained model from [Hugging Face](https://huggingface.co/WANGNingroci/VeriSeek), use the command 
```bash
huggingface-cli download --resume-download WANGNingroci/VeriSeek --local-dir VeriSeek
```
If you are located in China mainland, you can export the mirror backend to accelerate the download process.
```bash
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --resume-download WANGNingroci/VeriSeek --local-dir VeriSeek
```
The model will be downloaded to the `VeriSeek` directory.
Then modify the `config.json` under the `VeriSeek` directory, change the `architectures` to `["LlamaForCausalLMGrit"]` and `model_type` to `"llama_grit"`.

### Data Preparation
Download the training dataset from [Hugging Face](https://huggingface.co/datasets/WANGNingroci/BuggyVerilog/), and put the two files under `data/train` directory.
### Run Script
Run the training script with the following command:
```bash
bash scripts/train_llama_7b.sh
```
The model will be saved to the `output` directory.

### Post Processing
After training, you have to run 
```bash
python scripts/reformat_statedict.py ./output/pytorch_model.bin 
```
to remove the model. prefix from the checkpoint.

## Inference
If you want to infer the model, you can download the pre-trained model from [Hugging Face](https://huggingface.co/WANGNingroci/VeriDebug), use the command 
```bash
huggingface-cli download --resume-download WANGNingroci/VeriDebug --local-dir VeriDebug
```
And run the demo script with the following command:
```bash
python test.py
```

## Evaluation
### Line Location Accuracy@1, 3, 5
Run
```bash
python eval_line.py
```
### Type Accuracy@1, 3
Run
```bash
python eval_cls.py
```
### Generation Accuracy@1
To evaluate the generation accuracy, you can run the following command:
```bash
python eval_gen.py
```

## Citation