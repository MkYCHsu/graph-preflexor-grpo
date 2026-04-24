#!/bin/bash
#SBATCH --job-name="data_gen"
#SBATCH --partition=sched_mit_buehler
###SBATCH --partition=sched_mit_buehler_gpu
###SBATCH --gres=gpu:5
###SBATCH --gpu-bind=map_gpu:0,1,2,3,4

###SBATCH -N 4
#SBATCH -n 32
#SBATCH --mem-per-cpu=16G

#SBATCH --time=12:0:0
#SBATCH --output=cout.txt
#SBATCH --error=cerr.txt

###SBATCH --nodelist=node1229
###SBATCH --nodelist=node982

module purge
source ~/.bashrc
clean
#source ~/ml.sh
#conda deactivate
conda activate llm

XDG_RUNTIME_DIR=""

# python src/make_graph_dataset_advanced.py   --datasets "karpathy/fineweb-edu-100b-shuffle[:2000]|mkychsu/semiconductor_small"   --num_examples 2048   --teacher_model gpt-4o   --teacher_api_key $OPENAI_API_KEY   --reject_model gpt-4o-mini   --reject_api_key $OPENAI_API_KEY   --output_path ./graph_reasoning_advanced_tsmc.jsonl   --save_steps 100   --resume   --push_to_hub   --output_repo mkychsu/tsmc_small_preflexor_grpo

python src/make_graph_dataset_advanced_graphrag.py \
--datasets "karpathy/fineweb-edu-100b-shuffle[:2000]|mkychsu/semiconductor_small" \
--num_examples 2048 \
--teacher_model gpt-4o \
--teacher_api_key $OPENAI_API_KEY \
--reject_model gpt-4o-mini \
--reject_api_key $OPENAI_API_KEY \
--output_path ./graph_reasoning_advanced_tsmc.jsonl  \
--save_steps 100  \
--resume  \
--push_to_hub  \
--output_repo mkychsu/semiconductcor_preflexor_grpo_2048 \
--graph_rag_verbose \
--graph_rag_graphml_path "../GRAPHDATA_TSMC_OUTPUT/tsmc_5b10p.graphml" \
--graph_rag_chunk_dir "../GRAPHDATA_TSMC" \
--graph_rag_embedding_model_path "/home/mkychsu/pool/llm/SEMIKONG-8b-GPTQ" \
--graph_rag_embedding_cache_path "../GRAPHDATA_TSMC/TSMC_SEMIKONG.pkl"



