data_name=amazon_book
cuda='2'
lora_ckpt=lora_ckpt/${data_name}/base/checkpoint_best.pth
generate_seed=2000
test_num=2 
generate_num=80 
valid_num=1 
generate_synthetic_version=v0
save_synthetic_path=synthetic_data/${data_name}/tallrec_base/generate_num${generate_num}_for_data${test_num}_${generate_synthetic_version}_generate_seed_${generate_seed}.pkl
mkdir -p $(dirname ${save_synthetic_path})
output_dir=lora_ckpt/${data_name}/base/generate_num${generate_num}_for_data${test_num}
python TALLRec_main.py  --data_name ${data_name} --cuda $cuda\
        --test_num ${test_num} --valid_num $valid_num --batch_size_eval 12\
        --output_dir ${output_dir}\
        --evaluate_only 1 --generate_synthetic 1 --lora_ckpt ${lora_ckpt} --generate_seed $generate_seed\
        --generate_num ${generate_num} --save_synthetic_path ${save_synthetic_path} --generate_synthetic_version ${generate_synthetic_version}\

