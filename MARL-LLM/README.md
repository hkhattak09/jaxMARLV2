# MARL-LLM

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Ubuntu 20.04](https://img.shields.io/badge/ubuntu-20.04-orange.svg)](https://releases.ubuntu.com/20.04/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 🔍 Overview

This repository contains the code for our paper **LAMARL: LLM-Aided Multi-Agent Reinforcement Learning for Cooperative Policy Generation**. LAMARL consists of two main components: an LLM-aided automatic function generation module and a MARL module.

## ✨ Features
- 🤖 Automatic function generation using LLMs
- 🎯 Multi-agent reinforcement learning with cooperative policies
- 🚀 Accelerated environment sampling with C++ optimization
- 📊 Comprehensive evaluation and visualization tools
- 🔧 Modular and extensible architecture

## 📋 Requirements

- **Operating System**: Ubuntu 20.04 (recommended)
- **Python**: 3.10
- **GPU**: CUDA-compatible GPU
- 
## 📁 Project Structure

```
MARL-LLM/
├── marl_llm/                 # Main MARL-LLM implementation
│   ├── cfg/                  # Configuration files
│   ├── eval/                 # Evaluation scripts
│   ├── llm/                  # LLM modules
│   └── train/                # Training scripts
├── cus_gym/                  # Custom gym environment
│   ├── gym/                  # Gym environment implementation
│   └── ...
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## 🔧 Troubleshooting

## 🛠️ Installation

### 1. Create Virtual Environment
```bash
conda create -n marl_llm python=3.10
conda activate marl_llm
```

### 2. Install Dependencies
Navigate to the 'marl_llm' folder and install required packages:
```bash
cd marl_llm
pip install -r requirements.txt
```

### 3. Install PyTorch
Visit the [PyTorch official website](https://pytorch.org/get-started/previous-versions/) and install the GPU version according to your system configuration:
```bash
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
```

### 4. Install Custom Environment
Navigate to the 'cus_gym' folder and install the MARL-LLM environment:
```bash
cd ../cus_gym
pip install -e .
```

### 5. Set Environment Variables
Add the following path to your bashrc file:
```bash
echo 'export PYTHONPATH="$PYTHONPATH:/path/to/your/marl_llm/"' >> ~/.bashrc
source ~/.bashrc
```

### 6. Compile C++ Library
Some environment functions are implemented in C++ for acceleration:
```bash
cd cus_gym/gym/envs/customized_envs/envs_cplus
chmod +x build.sh
./build.sh
```

### 7. Configure LLM API
If you want to use LLM for reward function generation, configure your API credentials and run:
```bash
python ./marl_llm/llm/modules/framework/actions/rl_generate_functions.py
```

## 🚀 Usage

### Training

1. **Configure settings**: Set the variable `image_folder` in `cfg/assembly_cfg.py`:
   ```python
   image_folder = '/path/to/your/figures/'
   ```

2. **Start training**:
   ```bash
   cd marl_llm/train
   python train_assembly.py
   ```

### Evaluation

1. **Update experiment directory**: Copy the experimental directory name and replace `curr_run` in `eval_assembly.py`:
   ```python
   curr_run = '2025-01-19-15-58-03'  # Replace with your experiment timestamp
   ```

2. **Run evaluation**:
   ```bash
   python eval_assembly.py
   ```

### Common Issues

- **Missing packages**: Install them manually using `pip install package_name`
- **CUDA issues**: Ensure your GPU drivers and CUDA version are compatible with PyTorch
- **C++ compilation errors**: Make sure you have a compatible C++ compiler installed

### Getting Help

If you encounter any issues, please:
1. Check the [Issues](https://github.com/Guobin-Zhu/MARL-LLM/issues) page for existing solutions
2. Open a new issue with detailed error messages and system information

## 📝 Citation

If you find this code useful in your research, please consider citing our paper:

```bibtex
@article{zhu2025lamarl,
  author={Zhu, Guobin and Zhou, Rui and Ji, Wenkang and Zhao, Shiyu},
  journal={IEEE Robotics and Automation Letters}, 
  title={LAMARL: LLM-Aided Multi-Agent Reinforcement Learning for Cooperative Policy Generation}, 
  volume={10},
  number={7},
  pages={7476-7483},
  year={2025},
}

```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Note**: This documentation is continuously being updated. For the latest information, please check the repository regularly.