import argparse
from omegaconf import OmegaConf
import os

def parse_args():
    parser = argparse.ArgumentParser(description='configs')
    # baselines params
    parser.add_argument('--num_per_gpu',type = float,default = 0.1)
    parser.add_argument('--global_save',type = int,default = 1)
    parser.add_argument('--trial_name',type = str,default = '0111')
    parser.add_argument("--exp_name", default='base',type = str)
    parser.add_argument("--epoch", default=1000,type = int)
    parser.add_argument('--need_save_content',type=int,default=1)
    parser.add_argument("--load_model_exp_name", default='base',type = str)
    parser.add_argument("--if_load_model_exp_name", default='base',type = str)
    parser.add_argument("--if_load_model_exp_name0", default='base',type = str)
    parser.add_argument("--if_sampling", default=None,type = str)
    parser.add_argument('--cold_model_name',type = str,default = 'deepmusic')
    parser.add_argument('--tune_item_only',type = str,default = 'False')
    parser.add_argument('--iter_num',type = str,default = '0')
    parser.add_argument('--min_item_num',type = int,default = 1)
    parser.add_argument('--start_iter_num',type = str,default = '0')
    parser.add_argument('--end_iter_num',type = str,default = '0')
    parser.add_argument('--ensemble_k',type = float,default = 0.01)
    parser.add_argument('--perturb_ratio',type = float,default = 0.05)
    parser.add_argument('--if_movement',type = float,default = 1)
    parser.add_argument('--report_all',type = str,default = 'True')
    parser.add_argument('--cal_influence_on', choices=['valid', 'test'],type = str,default = 'valid')
    # parser.add_argument('--synthetic_save_name_for_influence', type=str, default=None) # synthetic data name
    parser.add_argument('--influence_save_name', type=str, default=None)  # save dict name
    parser.add_argument('--influence_save_name0', type=str, default=None)  # save dict name
    parser.add_argument('--if_test_data_path', type=str, default=None)  # save dict name
    # general file_path
    parser.add_argument("--data_dir", default=None,type = str)
    parser.add_argument("--workspace_name", default='workspace',type = str)   
    parser.add_argument("--llama_model", default='/data/xxx/LLM2024/llama2-7b-hf',type = str)

    # temp_param: damp, opt_lr, init_mul. 
    parser.add_argument('--init_mul',type = float,default = 0.1)
    parser.add_argument('--opt_lr',type = float,default = 0.1)
    parser.add_argument('--damp',type = float,default = 0.01)



    parser.add_argument("--rec_model", default="NCF",type = str)
    parser.add_argument("--data_name", default="amazon_book",type = str)

    # hyper-parameter
    parser.add_argument("--param_stage", default="stage1",type = str) 
    parser.add_argument("--freeze_stage", default="stage1",type = str)
    parser.add_argument("--prompt_stage", default="stage1",type = str)

    parser.add_argument("--prompt_version", default="v1",type = str)
    parser.add_argument("--dataset_version", default="v1",type = str) 
 
    parser.add_argument("--init_lr", default=None,type = float)
    parser.add_argument("--min_lr", default=None,type = float)
    parser.add_argument("--warmup_lr", default=None,type = float)
    parser.add_argument("--weight_decay", default=None,type = float)
    # load & evaluate & save
    # parser.add_argument("--path", default=None,type = str)
    # parser.add_argument("--dataset_dir", default=None,type = str)
    parser.add_argument("--output_dir", default=None,type = str) # begin from workspace
    parser.add_argument("--proj_ckpt", default=None,type = str)
    parser.add_argument("--lora_ckpt", default=None,type = str) 
    parser.add_argument("--pretrained_path", default=None,type = str)
    parser.add_argument("--if_pretrained_path", default=None,type = str)
    parser.add_argument("--save_test_path", default=None,type = str)
    parser.add_argument("--save_valid_path", default=None,type = str)
    parser.add_argument("--save_synthetic_path", default=None,type = str)
    parser.add_argument("--synthetic_data_path", default=None,type = str)
    parser.add_argument("--generate_synthetic_version", default='v0',type = str)
    parser.add_argument("--generate_seed", default=2024,type = int)
    parser.add_argument("--run_seed", default=2023,type = int)
    parser.add_argument("--mode", default='v3',type = str)
    parser.add_argument("--calculate_if_data_num", default=1,type = int) # z_test
    parser.add_argument("--calculate_if_train_data_num", default=0,type = int) # second term H theta. train
    # parser.add_argument("--calculate_if_z_data_num", default=1,type = int) # test{num} 
    parser.add_argument("--calculate_if_data_path", default=None,type = str)    # z_test 
    parser.add_argument("--dpo_test_num", default=2,type = int) # 



    # evaluate 
    parser.add_argument("--train_synthetic", default='False',type = str)
    parser.add_argument("--generate_synthetic", default='False',type = str)
    parser.add_argument("--evaluate_only", default='False',type = str)
    # parser.add_argument("--test_splits", nargs='+', default=None,type = str)

    # data 
    parser.add_argument("--train_start_num", default=0,type = int)
    parser.add_argument("--train_total_num", default=1,type = int)
    parser.add_argument("--valid_num", default=0,type = int)
    parser.add_argument("--test_num", default=0,type = int)
    parser.add_argument("--generate_num", default=None,type = int)

    # common 
    parser.add_argument("--iters_per_epoch", default=64,type = int)
    parser.add_argument("--batch_size_eval", default=4,type = int)
    parser.add_argument("--batch_size_train", default=16,type = int)
    parser.add_argument("--patience", default=6,type = int)
    parser.add_argument("--accum_grad_iters", default=1,type = int)
    parser.add_argument("--use_gradient_checkpoint", default='False',type = str)
    parser.add_argument("--use_logit_softmax", default='False',type = str)
    parser.add_argument("--use_8bit_optim", default='False',type = str)
    parser.add_argument("--cuda", default='0',type = str)
    
    return parser.parse_args()



config = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = config.cuda
lora_config_dict = {
    'r':8,
    'use_lora':True,
    'alpha':16,
    'target_modules':["q_proj", "v_proj"],
    'dropout':0.05
}

config.lora_config = OmegaConf.create(lora_config_dict)
config = vars(config) # convert to dict
config = OmegaConf.create(config) # convert to omegaconf dict
config.max_txt_len = 1024; config.proj_token_num = 1; config.proj_drop = 0; config.proj_mid_times = 10
config.end_sym = "###"; config.prompt_template = '{}'; config.user_num = -100; config.item_num = -100
config.ans_type = 'v2';  config.lora_config.use_lora = True; config.lora_config.r = 8; config.lora_config.alpha = 16
config.lora_config.target_modules = ["q_proj", "v_proj"]; config.lora_config.dropout = 0.05;  
config.num_workers = 0; config.amp = True
config.resume_ckpt_path = None; config.train_splits = ["train"]; config.valid_splits = ["valid"]; config.test_splits = ["test", "valid"]
config.synthetic_splits = ['synthetic']; config.device = "cuda"; config.world_size = 1; config.dist_url = "env://"; config.distributed = False
config.max_epoch = 200; config.warmup_steps = 200;config.seed = 42
config.evaluate_only = False if config.evaluate_only == 'False' else True
config.train_synthetic = False if config.train_synthetic == 'False' else True
config.generate_synthetic = False if config.generate_synthetic == 'False' else True
config.tune_item_only = False if config.tune_item_only == 'False' else True
config.report_all = False if config.report_all == 'False' else True
config.use_gradient_checkpoint = False if config.use_gradient_checkpoint == 'False' else True
config.use_8bit_optim = False if config.use_8bit_optim == 'False' else True
config.use_logit_softmax = False if config.use_logit_softmax == 'False' else True

if config.generate_synthetic:
    config.test_splits = []


# if 'llama2-7b' in config.llama_model:
config.llama_hidden_size = 4096

current_dir = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
config.workspace_dir = os.path.join(current_dir,config.workspace_name )
for dir in ['output_dir','proj_ckpt','lora_ckpt','pretrained_path','if_pretrained_path',
            'save_test_path','save_valid_path','save_synthetic_path','synthetic_data_path','if_test_data_path']:
    if config[dir] is not None:
        config[dir] = os.path.join(config.workspace_dir, config[dir]) # 填写格式 lora_ckpt / ..... / (一定不能从/开始)


if config.freeze_stage == 'stage1':
    config.freeze_rec = True   
    config.freeze_proj = True    
    config.freeze_lora = False 
elif config.freeze_stage == 'stage2':
    config.freeze_rec = True   
    config.freeze_proj = False   
    config.freeze_lora = True 
else:
    raise Exception('Not Implemented freeze stage')


arg_param_dict = {}
param_list = ['init_lr', 'min_lr', 'warmup_lr', 'weight_decay']
for param_name in param_list:
    arg_param_dict[param_name] = config.get(param_name, None) # 复制，即便前面无None也可以

if config.data_name == 'amazon_book': 
    config.user_num = 50000
    config.item_num = 50000
    config.category_num = 284 + 1
    config.text_embedding_size = 4096
    config.proj_mid = 5
    config.embedding_size = 128

    if config.param_stage == 'stage1':
        config.init_lr = 1e-4 
        config.min_lr = 1e-4  
        config.warmup_lr = 1e-4   
        config.weight_decay = 0  
    elif config.param_stage == 'stage2':
        config.init_lr = 1e-5
        config.min_lr = 1e-5 
        config.warmup_lr = 1e-5  
        config.weight_decay = 1e-3  
    else:
        raise Exception('Not Implemented running stage and corresponding parameters')

elif config.data_name == 'ml-32m': 
    config.item_num =  50000
    config.user_num = 50000
    config.embedding_size = 128
    config.proj_mid = 5
    config.category_num = 284 + 1
    config.text_embedding_size = 4096
    if config.param_stage == 'stage1':
        config.init_lr = 1e-4 # 1e-4
        config.min_lr = 1e-4 # 1e-4  
        config.warmup_lr = 1e-4 # 1e-4  
        config.weight_decay = 0  
    elif config.param_stage == 'stage2':
        config.init_lr = 1e-5
        config.min_lr = 1e-5 
        config.warmup_lr = 1e-5  
        config.weight_decay = 1e-3  
    else:
        raise Exception('Not Implemented running stage and corresponding parameters')

for param_name in param_list:
    param = arg_param_dict.get(param_name, None)
    if param is not None:
        config[param_name] = param  # 覆盖

if config.data_name == 'amazon_book': 
    if config.prompt_version == 'v1':
        if config.prompt_stage == 'stage1':
            config.prompt_path = 'prompts/tallrec_amazon.txt'
        elif config.prompt_stage == 'stage2':
            config.prompt_path = 'prompts/collm_amazon.txt'
        else:
            raise Exception('Not Implemented prompt stage')
    else:
        raise Exception('Not Implemented prompt version')
elif 'amazon' in config.data_name: # config.data_name == 'amazon_game': 
    if config.prompt_version == 'v1':
        if config.prompt_stage == 'stage1':
            config.prompt_path = f'prompts/tallrec_{config.data_name}.txt'
        elif config.prompt_stage == 'stage2':
            config.prompt_path = f'prompts/collm_{config.data_name}.txt'
        else:
            raise Exception('Not Implemented prompt stage')
    else:
        raise Exception('Not Implemented prompt version')
elif 'ml-32m' in config.data_name: # config.data_name == 'amazon_game': 
    if config.prompt_version == 'v1':
        if config.prompt_stage == 'stage1':
            config.prompt_path = f'prompts/tallrec_movie.txt'
        elif config.prompt_stage == 'stage2':
            config.prompt_path = f'prompts/collm_movie.txt'
        else:
            raise Exception('Not Implemented prompt stage')
    else:
        raise Exception('Not Implemented prompt version')

config.dataset_dir = os.path.join(config.data_dir, config.data_name) 
config.iid2content_dict_path = os.path.join(config.dataset_dir, 'iid2content_dict.pth')
config.iid2titleemb_dict_path = os.path.join(config.dataset_dir, 'iid2titleemb_dict.pth')
