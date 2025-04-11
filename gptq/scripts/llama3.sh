CUDA_VISIBLE_DEVICES=$1 python llama.py \
    meta-llama/Meta-Llama-3-8B \
    c4 \
    --wbits 4 \
    --true-sequential \
    --new-eval 

# use --save_in_16bits to save the model in 16-bit format
