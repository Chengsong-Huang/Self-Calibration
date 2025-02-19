# Efficient Test-Time Scaling via Self-Calibration

This repository contains the code and released models of our paper ```Efficient Test-Time Scaling via Self-Calibration```. We propose a new framework, Self-Calibration, that can make model generate calibrated confidence score. This calibrated confidence can make test-time scaling methods more efficient.

## Installation
```bash
conda create -n Self-Calibration python=3.10
conda activate Self-Calibration
pip install -r requirements.txt
pip install vllm -U

```
##Usage
### Data_Creation

```
bash scripts/data_gen.bash \
    meta-llama/Llama-3.1-8B-Instruct \
    0.8 \
    "--use_cot" \
    32 \
    "train" \
    100 \
    "llama"
```


```
bash scripts/main.bash ./models/llama llama meta-llama/Llama-3.1-8B-Instruct

```