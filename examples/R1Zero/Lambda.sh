#!/bin/bash

### To use the Lambda server
# 1. You should create a SSH Keys. The first time you create this, a file .pem will be downloaded to your computer. Please always use this pem file.

# 2. Ensure that the SSH key used during creating the Lambda’ instance is exactly the one you 1) have the pem file and 2) want to log in with.

# 3. Ensure that the SSH key (you can copy from the Lambda’s website) has been added: $ echo 'PUBLIC-KEY' >> ~/.ssh/authorized_keys
# where 'PUBLIC-KEY' here is your SSH key.

# 4. ssh -i '<SSH-KEY-FILE-PATH>' ubuntu@<INSTANCE-IP>
# where '<SSH-KEY-FILE-PATH>' should be your .pem file and ‘<INSTANCE-IP>’ can be found on the crated instance on the website.



# login in 
ssh -i /Users/sjia/Documents/PersonalCloud/Lambda/SjiaLambda.pem ubuntu@129.158.245.28


# Copy from the server to the local
$ scp -i /Users/sjia/Documents/PersonalCloud/Lambda/SjiaLambda.pem -r ubuntu@150.136.213.72:/home/ubuntu/Sjia/acl25-plan-code/ExptsPlanFT .

$ scp -i /Users/sjia/Documents/PersonalCloud/Lambda/SjiaLambda.pem -r ubuntu@150.136.213.72:/home/ubuntu/Sjia/acl25-plan-code/wandb .


# Copy a checkpoint from the local to the server
$ scp  -i /Users/sjia/Documents/PersonalCloud/Lambda/SjiaLambda.pem -r /Users/sjia/Documents/Research/MyPapers/LLMReasoningPlanMemory/repo/experiments/learner-checkpoints/ConceptLearner--all-MiniLM-L6-v1--512--Qwen2.5-0.5B-Instruct--finetuned--direct-gpt4o-MATH--implicit-latentPlan/run_2025-02-10/checkpoint-6408  ubuntu@150.136.213.72:/home/ubuntu/Sjia/acl25-plan-code/ICMLPlan/checkpoints/

$ scp  -i /Users/sjia/Documents/PersonalCloud/Lambda/SjiaLambda.pem /Users/sjia/Documents/Research/MyPapers/LLMReasoningPlanMemory/repo/acl25-plan-code/examples/LatentPlan/cloud_init.sh ubuntu@150.136.213.72:/home/ubuntu/Sjia

$ scp  -i /Users/sjia/Documents/PersonalCloud/Lambda/SjiaLambda.pem /Users/sjia/Documents/Research/MyProjects/dmmrl/.env ubuntu@129.158.245.28:/home/ubuntu/Sjia/dmmrl


# Git
# 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

$ git clone https://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@github.com/CSJDeveloper/dmmrl.git

$ git clone 

kill -9 7340 


