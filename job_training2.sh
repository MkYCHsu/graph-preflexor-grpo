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

python src/run_grpo_graph.py \
  --base_model_dir mkychsu/semiconductor_graph_preflexor_grpo \
  --dataset mkychsu/semiconductcor_preflexor_grpo_2048 \
  --output_dir ./orpo-grpo-graph_v1 \
  --judge_model gpt-5-mini \
  --judge_api_key $OPENAI_API_KEY \
  --epochs 1 \
  --num_generations 2 \
  --push_to_hub \
  --hub_model_id mkychsu/semiconductor_graph_preflexor_grpo_2 \
  --hf_token $HF_TOKEN

