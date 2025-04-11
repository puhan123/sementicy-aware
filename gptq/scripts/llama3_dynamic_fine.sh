CUDA_VISIBLE_DEVICES=$1 python llama.py \
    $4 \
    c4 \
    --wbits 4 \
    --true-sequential \
    --new-eval \
    --fine-wbits-yaml configs/$2.yaml \
    --save_in_16bits $3 \
    --no-eval \
    --important_mask $5