# dmmrl: A Lightweight, Easy-to-Use Platform for Fine-tuning LLMs and VLMs via RL Algorithms

![image](https://github.com/CSJDeveloper/open-dmmrl/blob/main/Images/logo-main.jpg) 


> ### Roadmap for dmmrl
> We aim to simplify experiments using RL to enhance the step-by-step reasoning abilities of large foundation models on tasks involving text and multimodal datasets. Our target is to provide a clean, organized, and user-friendly code structure to help users understand RL-based large model training and fine-tuning, thereby facilitating learning and personal usage.
> 
> > Short-term: Evaluate the effectiveness of the __GRPO__ algorithm across various unimodal and multimodal datasets.
> > 
> > Long-term: Enable straightforward training and fine-tuning of diverse LLMs, VLMs, and MLLMs using a variety of RL algorithms across multiple tasks and scenarios.
>
> Benchmark-purpose: We aim to provide a benchmark for fairly comparing different combinations of RL algorithms with LLMs, VLMs, and MLLMs for improving their reasoning abilities.
>
> Welcome Ideas and Contribution. Stay tuned!


---

### Updates

- _[TODO]_ `2025-03-22: We will release more advanced mechanisms for adjusting the attention of the VLMs.`.
- _[TODO]_ `2025-03-18: We will release most of our results on both LLMs and VLMs.`.
- 2025-03-16: Released the full codebase.
- 2025-03-16: Uploaded checkpoints, some experimental results, and demos.
- 2025-03-08: Added a visual example demonstrating how Qwen-2-VL-72B trained with GRPO effectively performs step-wise reasoning across both textual and image modalities.
- 2025-03-08: Uploaded README.md and included detailed comments throughout the codebase to help users easily understand key operations and the logic within each component.
- 2025-03-08: We released a partial codebase, along with selected configurations and examples for training LLMs and VLMs using GRPO across four benchmark datasets.
  

## Setup

```bash
conda create -n RLReasoning python=3.11 
conda activate RLReasoning

pip install .
```

## Code structure
The structure of `llmpebase` is 

    .
    ├── configs                         # Configuration files to be used
    ├── examples                        # Implemented approaches, such as R1-Zero
    ├── dmmrl                           # The source code of `dmmrl` platform
    └──── dataset                        # Available Datasets
    └──── rewarder                       # Available Reward functions 
    └──── tools                          # Useful tools to simplify coding
    └──── trainer                        # RL-based Training pipeline 


### Supported Models

Models marked with `*` indicate availability in various types (e.g., instruct, base) and sizes (e.g., 3B, 7B).

- Large language models (LLMs)
  - Qwen2-*
  - Qwen2.5-*
  - Llama3.*
  
- Multimodal large language models (MLLMs), including Visual language models (VLMs)
  - Qwen2-VL-*
  - Qwen2.5-VL-*



### Supported Datasets 

Toward datasets denoting as language (L), visual-language (VL) for :school_satchel: training and :mortar_board: evaluation, we have: 

1. [L, MATH]    [GSM8K](https://huggingface.co/datasets/openai/gsm8k). -> :school_satchel: :mortar_board:
   
3. [L, MATH]    [MATH-lighteval](https://huggingface.co/datasets/DigitalLearningGmbH/MATH-lighteval). -> :school_satchel: :mortar_board:
   
4. [L, MD]    [TheoremQA](https://github.com/wenhuchen/TheoremQA). -> :mortar_board:
   
5. [VL, MD]    [MMMU](https://huggingface.co/datasets/lmms-lab/MMMU). -> :mortar_board:
   
6. [VL, MD]    [ScienceQA](https://huggingface.co/datasets/armanc/ScienceQA). -> :school_satchel: :mortar_board:

where `MD` denotes Multidisciplinary.

## Training

To run the code, please refer to the `examples/` directory for available usage examples and the `configs/` directory for corresponding configuration files.

A common command line to run:

`python <YourApproach>.py -c <Configuration> -b <ExperimentName> -r <ExperimentRecordFilename>`


### GRPO

- GSM8K: 
  - ```console
    accelerate launch --config_file configs/AccelerateConfigs/zero3.yaml python examples/R1Zero/R1Zero.py -c configs/GSM8K/R1Zero/Qwen2.5-0.5B.yml -b Experiment -p Experiment -r experiments.csv 
    ```

- MATH: 
  - Single GPU:  
    ```console
    python examples/R1Zero/R1Zero.py -c configs/MATH/R1Zero/Qwen2.5-0.5B.yml -b Experiment -p pR1Zero -r experiments.csv 
    ```
  - Multiple GPU: 
    ```console
    accelerate launch --config_file configs/AccelerateConfigs/zero3.yaml examples/R1Zero/R1Zero.py -c configs/MATH/R1Zero/Qwen2.5-0.5B.yml -b Experiment -p pR1Zero -r experiments.csv 
    ```

- ScienceQA: 
  - Single GPU:  
    ```console
    python examples/R1Zero/R1ZeroVL.py -c configs/ScienceQA/R1Zero/Qwen2.5-vl-3b-instruct-TEST.yml -b Experiment -p pR1ZeroVL -r experiments.csv 
    ```
  - Multiple GPU: 
     ```console
    accelerate launch --config_file configs/AccelerateConfigs/zero3.yaml python examples/R1Zero/R1ZeroVL.py -c configs/ScienceQA/R1Zero/Qwen2.5-vl-3b.yml -b Experiment -r experiments.csv 
    ```

## Demo

We allow the Qwen2.5-VL-72B to answer the question: _Refer to the Figure. Determine the theoretical maximum area of the Earth's surface that would be in view from a geostationary satellite orbiting at a height of 35786 km from the surface of the Earth. Also determine the area in view for a minimum elevation angle of \(10^{\circ}\). (Assume that the radius of the Earth is 6378 km.)_

![image](https://github.com/CSJDeveloper/open-dmmrl/blob/main/Images/demos/simple-demo.png)

## Results

### "Aha" moment does not appear in the weak LLMs

We do not observe an "Aha" moment in Qwen2.5-0.5B with batch_size =4 and group_size=4.
  <details>
  <summary><strong>Training Details</strong></summary>
    
  ![image](https://github.com/CSJDeveloper/open-dmmrl/blob/main/Images/results/Qwen2.5-0.5B.png) 
  
  </details>

After increasing the batch size to 16 and the number of generations (G in GROP) to 16, the "Aha" moment still does not appear.
  <details>
  <summary><strong>Training Details</strong></summary>
    
  ![image](https://github.com/CSJDeveloper/open-dmmrl/blob/main/Images/results/Qwen2.5-0.5-Batchsize16-G16.png) 
  
  </details>



### "Aha" moment gradually appear in stronger LLMs

We observe that when using Qwen2.5-3B and Qwen2.5-7B as the base models, their accuracy increases significantly with GRPO training. However, there is not a large performance gap between these results and those obtained through supervised fine-tuning.



## Acknowledgements

We sincerely thank [DeepSeek](https://github.com/deepseek-ai/DeepSeek-R1), [Open-R1](https://github.com/huggingface/open-r1), [QwenVL](https://github.com/QwenLM/Qwen2.5-VL), [VLM-R1](https://github.com/om-ai-lab/VLM-R1), [R1-V](https://github.com/Deep-Agent/R1-V) for their valuable open-source code, which provided substantial support in building our platform.


## Potential Packages
- PyMARL, EPyMARL (MADDPG/MATD3 supported)
- PettingZoo (multi-agent RL environments)
- RLlib (Ray) (strongly supports MADDPG, MATD3, FACMAC)
- Stable-Baselines3 (single-agent RL—can be manually extended for multi-agent)

- Stackelberg MARL (Leader-Follower)
- Hierarchical RL

## Citation

```bib
@misc{chen2025dmmrl,
  author       = {Chen, Sijia},
  title        = {dmmrl: A Lightweight, Easy-to-Use Platform for Fine-tuning LLMs and VLMs via RL Algorithms},
  howpublished = {\url{https://github.com/CSJDeveloper/dmmrl}},
  note         = {Accessed: 2025-03-08},
  year         = {2025}
}
```










