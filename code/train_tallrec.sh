

data_name='amazon_book'
cuda='7'

learning_rate=1e-3
exp_name=base
output_dir=lora_ckpt/${data_name}/${exp_name}/  
python TALLRec_main.py --output_dir $output_dir  --cuda $cuda --data_name $data_name --mode v3\
        --min_lr ${learning_rate} --init_lr ${learning_rate} --warmup_lr ${learning_rate} 


####################### eval
for test_num in 2 1 0 # 0 1 2 # 0 1 2 
do
lora_ckpt=lora_ckpt/${data_name}/${exp_name}/checkpoint_best.pth

save_test_path=test_results/${data_name}/tallrec_${exp_name}/test${test_num}.json
python TALLRec_main.py --cuda $cuda --data_name $data_name\
        --save_test_path $save_test_path --evaluate_only 1 --lora_ckpt $lora_ckpt --test_num $test_num  --output_dir $output_dir\
        --batch_size_eval 4 --valid_num $test_num
done



