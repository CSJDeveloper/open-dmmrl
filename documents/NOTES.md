# Note Obtained during Experiments


### Implementation of the GRPO

1. Several important things about the implementation of the GRPO presented in the [DeepSeekMath](https://arxiv.org/pdf/2402.03300) are that:
    - the `sync_ref_model` hyper-parameter is set to be False by default
    - because The reference model is only updated at the beginning of each epoch. And when we finetune the model for one epoch, there is no need to sync the reference model. (But you absolutely should consider this when you train the model for multiple epochs or need other operations related to the reference model.)
    - There is a model with the policy_old in the algorithm to support the multiple iterations (u in the paper) within each batch of examples.

2. This implementation is not strictly aligned with the one presented in the paper of GRPO because we do not contain the number of iteration `u`.

3. Note that the `batch_size`, and `num_generations` is: 
    > total_batch_size_across_processes = batch_size * num_processes

    Then, as each sample will be repeated by `num_generations` times and the number of samples will selected should be digital numbers. It is necessary to ensure:
    
    > total_batch_size_across_processes = #samples * num_generations



### Training of the GRPO

1. It is completely normal for the loss to start at zero and then increase (loss = 0.0). See the discussion in the [issue](https://github.com/huggingface/open-r1/issues/239#issuecomment-2646297851).


##### Qwen/Qwen2.5-0.5B

1. In the original implementation, using `per_device_train_batch_size=4, num_generations=4`, training runs on a single 30GB A100 GPU. Initially, both `accuracy_reward` and `format_reward` are 0; however, over the first few batches, `format_reward` quickly increases to values like 0.5 and 1, while `accuracy_reward` remains 0 in most batches. The corresponding command is ```python examples/R1Zero/R1Zero.py -c configs/MATH/R1Zero/Qwen2.5-0.5B-batch4-G4.yml -b Experiment -p pR1Zero -r experiments.csv```

2. In another test, I introduced a short answer format instruction in the question prompt to force the model to place the final solution in a specific format. For example, for math problems, I used: `f"{problem} (Place final solution within \\boxed{{...}})"`. Despite this adjustment, the same trend as described in point 1 was observed.