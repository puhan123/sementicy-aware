#!/bin/bash
echo "Script Starting..."
source tacq_venv/bin/activate
set -x

importances_dir="/importances_dir/results"
results_dir="/eval_dir/results"
checkpoints_dir="/model_checkpoints"

model_name="Meta-Llama-3-8B-Instruct"
loadstring="meta-llama"
scoring_method="mean_abs"
ci_type="bootstrap"
ci_alpha="0.1"
ci_samples="256"
datasets=("c4_new")
serial_numbers=(0)
selector_types=("sample_abs_weight_prod_contrastive")
device="0,2,3"
eval_device="0"
quantization_types=("q2" "q3") 
ranking_types=("top_p_sparse") 
ratios=(".0035") # "0.001"

#Initiate an empty list to store all the eventual quantized model names
declare -a quantized_model_names

# Compute Intensive Section--3GPUs for 7-8B LLMs, efficiently capture importances for each weight.
for serial_number in "${serial_numbers[@]}"
do
    for cur_dataset in "${datasets[@]}"
    do
        dataset=$cur_dataset
        calibration_dataset=$cur_dataset
        for ratio in "${ratios[@]}"
        do  
            for ranking_type in "${ranking_types[@]}"
            do
                for quantization_type in "${quantization_types[@]}"
                do
                    wbits=${quantization_type%%_*}  # remove anything from first underscore onward
                    wbits=${wbits#q}               # remove the leading 'q'
                    echo "wbits: $wbits"
                        for selector_type in "${selector_types[@]}"
                        do     
                        corrupt_model_name="${model_name}+${serial_number}+${dataset}+${wbits}bit+quantized_model"
                        if [ ! -f "$checkpoints_dir/${corrupt_model_name}.pt" ]; then
                            echo "\n\nRunning gptq with run_name: ${corrupt_model_name}"
                            CUDA_VISIBLE_DEVICES=$device python -m gptq.llama \
                                $loadstring/${model_name} \
                                $dataset \
                                --true-sequential \
                                --save_in_16bits $checkpoints_dir/${corrupt_model_name}.pt \
                                --wbits $wbits \
                                --seed $serial_number \
                                --no-eval
                        fi

                        run_name="${model_name}+${serial_number}+${dataset}+${selector_type}+${wbits}bit+implementation_test"
                        echo "\n\nRunning measure_importances with run_name: ${run_name}"
                        CUDA_VISIBLE_DEVICES=$device python -m measure_importances \
                            --model $model_name \
                            --corrupt_model $corrupt_model_name \
                            --dataset $dataset \
                            --run_name $run_name \
                            --checkpoints_dir $checkpoints_dir \
                            --results_dir $importances_dir \
                            --selector_type $selector_type \
                            --serial_number $serial_number \
                            --save_full_gradients \
                            --save_importances_pt_path $importances_dir/$run_name/importances.pt \
                            --scoring_method $scoring_method \
                            --ci_type $ci_type \
                            --ci_alpha $ci_alpha \
                            --ci_samples $ci_samples \
                            --save_kappa_json_path $importances_dir/$run_name/kappa_scores.json \
                            --override_args_yaml \
                            --plot_importances
                            
                        rm -rf $checkpoints_dir/${corrupt_model_name}.pt
                    done
                done
            done
        done
    done
done

# Compute Low Section--1GPU for 7-8B LLMs, quantize based on weight importances and evaluate.
for serial_number in "${serial_numbers[@]}"
do
    for cur_dataset in "${datasets[@]}"
    do
        dataset=$cur_dataset
        calibration_dataset=$cur_dataset
        for ratio in "${ratios[@]}"
        do  
            for ranking_type in "${ranking_types[@]}"
            do
                for quantization_type in "${quantization_types[@]}"
                do
                    wbits=${quantization_type%%_*}  # remove anything from first underscore onward
                    wbits=${wbits#q}               # remove the leading 'q'
                    echo "wbits: $wbits"
                    for selector_type in "${selector_types[@]}"
                    do
                        corrupt_model_name="${model_name}+${serial_number}+${dataset}+${wbits}bit+quantized_model"
                        run_name="${model_name}+${serial_number}+${dataset}+${selector_type}+${wbits}bit+implementation_test"
                        echo -e "\n\nRunning make_quantization_configs with run_name: ${run_name}"
                        quant_identifier="${quantization_type}+${ranking_type}+${ratio}"
                        CUDA_VISIBLE_DEVICES=$eval_device python -m make_quantization_configs \
                            --run_name $run_name \
                            --checkpoints_dir $checkpoints_dir \
                            --results_dir $importances_dir \
                            --serial_number $serial_number \
                            --importances_pt_path $importances_dir/$run_name/importances.pt \
                            --mask_save_path $importances_dir/$run_name/important_mask_${quant_identifier}.pt \
                            --model $model_name \
                            --quantization_type $quantization_type \
                            --ranking_type $ranking_type \
                            --configs_save_path $importances_dir/$run_name/quantization_configs_${quant_identifier}.yaml \
                            --mask_fraction $ratio \
                            --proportional_total_params \
                            --force_recompute

                        quantized_model_name="${run_name}+gptq_on_${dataset}+${quant_identifier}+quantized_model"
                        echo -e "\n\nRunning gptq with run_name: ${quantized_model_name}"
                        CUDA_VISIBLE_DEVICES=$eval_device python -m gptq.llama \
                            $loadstring/${model_name} \
                            $dataset \
                            --true-sequential \
                            --fine-wbits-yaml $importances_dir/$run_name/quantization_configs_${quant_identifier}.yaml \
                            --save_in_16bits $checkpoints_dir/${quantized_model_name}.pt \
                            --no-eval \
                            --seed $serial_number \
                            --important_mask $importances_dir/$run_name/important_mask_${quant_identifier}.pt


                        valid_datasets=("MMLU_MCQA" "MMLU_STEM" "MMLU_humanities" "MMLU_social_sciences")
                        for valid_dataset in "${valid_datasets[@]}"
                        do
                            echo "\n\nRunning MMLU_Zero_Shot_Dataset eval with run_name: ${quantized_model_name}"
                            CUDA_VISIBLE_DEVICES=$eval_device python3 -m datasets_directory.MMLU.MMLU_eval \
                                --engine "${quantized_model_name}" \
                                --ntrain 5 \
                                --data_dir "datasets_directory/MMLU/data" \
                                --save_dir "$results_dir/$calibration_dataset" \
                                --addition_dir $checkpoints_dir \
                                --device "cuda" \
                                --MMLU_split $calibration_dataset \
                                --serial_number $serial_number
                        done


                        echo "\n\nRunning GSM8k eval with run_name: ${quantized_model_name}"
                        CUDA_VISIBLE_DEVICES=$eval_device python3 -m datasets_directory.GSM8k.GSM8k_eval \
                            --model_name_or_path ${quantized_model_name} \
                            --output_dir "$results_dir/GSM8k/${quantized_model_name}" \
                            --data_root "datasets_directory/GSM8k/data" \
                            --seed $serial_number \
                            --checkpoints_dir $checkpoints_dir


                        tables="datasets_directory/Spider/data/spider/tables.json"
                        dataset_path="datasets_directory/Spider/data/spider/dev.json"
                        dev_gold_path="datasets_directory/Spider/data/spider/dev_gold.sql"
                        db_dir="datasets_directory/Spider/database"
                        output_savedir="$results_directory/results/Spider/${quantized_model_name}"

                        CUDA_VISIBLE_DEVICES=$eval_device python -m datasets_directory.Spider.Spider_eval \
                            --model ${quantized_model_name} \
                            --input $dataset_path \
                            --tables $tables \
                            --predictions_filename predictions.txt \
                            --output "${output_savedir}/debug.txt" \
                            --output_savedir $output_savedir

                        export NLTK_DATA="datasets_directory/Spider/third_party/nltk_data"
                        python datasets_directory/Spider/third_party/test-suite-sql-eval/evaluation.py \
                            --gold $dev_gold_path \
                            --pred ${output_savedir}/predictions.txt \
                            --db $db_dir \
                            --table $tables \
                            --etype exec \
                            --output_savedir $output_savedir


                        perplexity_datasets=("wikitext2" "c4_new" "ptb_new")
                        for perplexity_dataset in "${perplexity_datasets[@]}"
                        do
                        CUDA_VISIBLE_DEVICES=$eval_device python3 -m datasets_directory.pretrain_datasets.perplexity_eval_16_bit \
                            --model "${quantized_model_name}" \
                            --save_dir "$results_dir/${perplexity_dataset}/${quantized_model_name}" \
                            --addition_dir $checkpoints_dir \
                            --device "cuda" \
                            --serial_number $serial_number \
                            --dataset_name $perplexity_dataset
                        done


                        quantized_model_names+=("$quantized_model_name")
                        rm -rf $checkpoints_dir/${quantized_model_name}.pt
                        rm -rf $importances_dir/$run_name/important_mask_${quant_identifier}.pt
                    done
                done
            done
        done
    done
done
echo "Quantized model names: ${quantized_model_names[@]}"
