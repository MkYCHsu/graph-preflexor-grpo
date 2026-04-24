#!/bin/bash
#SBATCH --job-name="train1"
###SBATCH --partition=sched_mit_buehler
#SBATCH --partition=sched_mit_buehler_gpu
#SBATCH --gres=gpu:1
###SBATCH --gpu-bind=map_gpu:0,1,2,3,4

###SBATCH -N 4
#SBATCH -n 32
#SBATCH --mem-per-cpu=16G

#SBATCH --time=12:0:0
#SBATCH --output=cout.txt
#SBATCH --error=cerr.txt

#SBATCH --nodelist=node1229
###SBATCH --nodelist=node982

module purge
source ~/.bashrc
~/clean_trash.sh
#source ~/ml.sh
#conda deactivate
conda activate llm

XDG_RUNTIME_DIR=""

#jupyter nbconvert --to script graphRAG_preflexor_v3-training.ipynb
#python graphRAG_preflexor_v3-training.py

python src/run_orpo_graph.py \
  --base_model ~/pool/llm/Llama-3.1-8B-Instruct \
  --dataset mkychsu/semiconductcor_preflexor_grpo_2048  \
  --output_dir ./orpo_graph_model \
  --lora_r 16 \
  --lora_alpha 32 \
  --lr 5e-5 \
  --epochs 1 \
  --batch_size 1 \
  --grad_accum 4 \
  --max_length 6144 \
  --save_steps 500 \
  --eval_steps 500 \
  --push_to_hub \
  --hub_model_id mkychsu/tsmc_graph_preflexor_grpo \
  --hf_token $HF_TOKEN


