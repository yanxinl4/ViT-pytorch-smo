set -e

python eval.py \
  --checkpoint_path output/cifar100-t-g-block_checkpoint.bin \
  --model_type ViT-T_16 \
  --dataset cifar100 \
  --eval_batch_size 256\
  --seed 42 \
  --smo \
  --save_file new-vit-block

