# Single GPU
export CUDA_VISIBLE_DEVICES=0
python run.py --config_path ./config/KELLER.yaml

# Multi GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun --nproc_per_node 8 run.py --config_path ./config/KELLER.yaml