import argparse
import os
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import pandas as pd
import numpy as np
import torch.optim
from sklearn.metrics import roc_auc_score
import torch.nn as nn
import torch.nn.functional as F
import omegaconf
import random 
import time
import numpy as np
import functools
import ray
from ray import train, tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from rec_models import MatrixFactorization,LightGCN,GnnDataset, NCF
from config import config
from torch.func import functional_call, vmap, hessian
from tqdm import tqdm
data_dir = config.data_dir

def create_folders(filename):
    if filename == None:
        return
    parent_folder = os.path.abspath(filename)
    while not os.path.exists(parent_folder):
        os.makedirs(parent_folder)
        print(f"Created parent folder '{parent_folder}'")
        parent_folder = os.path.dirname(parent_folder)

load_dir = os.path.join(config.workspace_dir,f"pretrained_models/{config.data_name}/{config.rec_model}/{config.if_load_model_exp_name}")

assert config.influence_save_name, "u should provide the save name"
influence_save_dir = os.path.join(config.workspace_dir,f"influence_function/{config.data_name}/{config.rec_model}/")
def create_folders(filename):
    if filename == None:
        return
    parent_folder = os.path.abspath(filename)
    while not os.path.exists(parent_folder):
        os.makedirs(parent_folder)
        print(f"Created parent folder '{parent_folder}'")
        parent_folder = os.path.dirname(parent_folder)
create_folders(influence_save_dir)

# set random seed   
def run_a_trail(train_config,  load_dir = None, save_test_path = None):
    seed=2023
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    all_data =[]
    full_train_list = list(range(config.train_start_num, config.train_start_num + config.train_total_num))
    train_data_paths = [os.path.join(data_dir,f'{config.data_name}/train{str(num)}.pkl') for num in full_train_list]
    val_data_paths = [os.path.join(data_dir,f'{config.data_name}/valid{str(num)}.pkl') for num in full_train_list if num != full_train_list[-1]]
    train_data_paths.extend(val_data_paths)
    for csv_file_path in train_data_paths:
        data = pd.read_pickle(csv_file_path)[['uid','iid','label']] # .values
        all_data.append(data)
    train_data = pd.concat(all_data, ignore_index=True)
    valid_path = os.path.join(data_dir,f'{config.data_name}/valid{str(config.valid_num)}.pkl')
    test_path = os.path.join(data_dir,f'{config.data_name}/test{str(config.test_num)}.pkl')
    valid_data = pd.read_pickle(valid_path)[['uid','iid','label']]
    test_data = pd.read_pickle(test_path)[['uid','iid','label']]

    if config.synthetic_data_path is not None:
        if os.path.exists(config.synthetic_data_path):
            synthetic_data = pd.read_pickle(config.synthetic_data_path)
            # convert logits to 0,1 label
            train_p1 = float(train_data['label'].mean())
            best_thre = 0
            best_diff = 100
            for thre in list(np.linspace(0,5,100)):
                predict_p1 = synthetic_data[synthetic_data['predicted_logit']>=thre].shape[0] / synthetic_data.shape[0]
                if abs(train_p1 - predict_p1) < best_diff:
                    best_diff = abs(train_p1 - predict_p1) 
                    best_thre = thre
            synthetic_data['label'] = np.where(synthetic_data['predicted_logit']>=best_thre,1,0 )
            synthetic_data = synthetic_data[['uid','iid','label']]
            # train_data = pd.concat([train_data,synthetic_data])

    user_num = config.user_num
    item_num = config.item_num
    print("user nums:", user_num, "item nums:", item_num)

    if config.data_name == 'ml-32m'  and (config.rec_model == 'lightgcn'):
        train_data = train_data.sample(frac = 0.3, random_state=1000)
    elif config.data_name == 'ml-32m'  and (config.rec_model == 'NCF'):
        train_data = train_data.sample(frac = 0.3, random_state=2000)
    train_data_loader = DataLoader(train_data.values, batch_size = train_data.shape[0], shuffle=False)
    synthetic_data_loader = DataLoader(synthetic_data.values, batch_size = 1, shuffle=False)


    if config.rec_model == 'MF':
        mf_config={
            "user_num": int(user_num),
            "item_num": int(item_num),
            "embedding_size": int(config['embedding_size'])
            }
        mf_config = omegaconf.OmegaConf.create(mf_config)

        model = MatrixFactorization(mf_config).cuda()
    elif config.rec_model == 'lightgcn':
        lgcn_config={
            "user_num": int(user_num),
            "item_num": int(item_num),
            "embedding_size": int(config['embedding_size']),
            "embed_size": int(config['embedding_size']),
            "dataset": config.data_name,
            "layer_size": '[64,64]',
            # lightgcn hyper-parameters
            "gcn_layers": 2, # TODO config
            "keep_prob" : 0.6,
            "A_n_fold": 100,
            "A_split": False,
            "dropout": False,
            "pretrain": 0,
            "init_emb": 1e-3, # TODO config
            }
        lgcn_config = omegaconf.OmegaConf.create(lgcn_config)
        gnndata = GnnDataset(lgcn_config, train = train_data, valid=valid_data,test=test_data, 
                             path = data_dir,m_users=user_num,n_item=item_num)
        model = LightGCN(lgcn_config).cuda()
        model._set_graph(gnndata.Graph)

    elif config.rec_model == 'NCF':
        print(config['embedding_size'])
        ncf_config={
            "user_num": int(user_num),
            "item_num": int(item_num),
            "embedding_size": int(config['embedding_size']),
            "dataset": config.data_name, 
            "hidden_size": 64,
            "init_emb": 1e-3,
            }
        ncf_config = omegaconf.OmegaConf.create(ncf_config)
        model = NCF(ncf_config).cuda()
    model_path = os.path.join(load_dir, 'best_model.pt')
    model.load_state_dict(torch.load(model_path))
    influence_score_dict = {}

    valid_data_loader = DataLoader(valid_data.values, batch_size = valid_data.shape[0], shuffle=False)
    test_data_loader = DataLoader(test_data.values, batch_size = test_data.shape[0], shuffle=False)
    if config.if_test_data_path is not None:
        if os.path.exists(config.if_test_data_path):
            calculate_influence_data = pd.read_pickle(config.if_test_data_path)
        else:
            raise Exception('influence function path do not exist')
            
        if 'label' not in calculate_influence_data.columns:
            if 'predicted_logit' not in calculate_influence_data.columns:
                print('data do not have labels and predicted logit')
            else:
                train_p1 = float(train_data['label'].mean())
                best_thre = 0
                best_diff = 100
                for thre in list(np.linspace(0,5,100)):
                    predict_p1 = calculate_influence_data[calculate_influence_data['predicted_logit']>=thre].shape[0] / calculate_influence_data.shape[0]
                    if abs(train_p1 - predict_p1) < best_diff:
                        best_diff = abs(train_p1 - predict_p1) 
                        best_thre = thre
                calculate_influence_data['label'] = np.where(calculate_influence_data['predicted_logit']>=best_thre,1,0 )
        calculate_influence_data = calculate_influence_data[['uid','iid','label']]
        calculate_influence_data_loader = DataLoader(calculate_influence_data.values, batch_size = 1000, shuffle=False)
    else:
        if config.cal_influence_on == 'valid':
            calculate_influence_data_loader = valid_data_loader # test_data_loaderloss
        elif config.cal_influence_on == 'test':
            calculate_influence_data_loader = test_data_loader # test_data_loader
        else:
            raise Exception('Not implemented calculate influence on')

    ############ CG WITH GPU 
    # STEP 1 INIT PARAMS
    # step 1.1 calculate test loss
    test_loss = 0
    for  batch_data in calculate_influence_data_loader:
        batch_data = batch_data.cuda().squeeze() #  (3)  uid, iid, label
        output = model(batch_data[:,0].long(), batch_data[:,1].long())
        label = batch_data[:,-1].float() # 1
        test_loss = torch.nn.functional.binary_cross_entropy_with_logits(output,label, reduction = 'mean') 
        # test_loss += loss
    # test_loss /= len(calculate_influence_data_loader)
    # step 1.2 calculate train loss
    torch.cuda.empty_cache()
    train_loss = 0
    for batch_data in train_data_loader:
        batch_data = batch_data.cuda().squeeze() #  (3)  uid, iid, label
        output = model(batch_data[:,0].long(), batch_data[:,1].long())
        label = batch_data[:,-1].float() # 1
        train_loss = torch.nn.functional.binary_cross_entropy_with_logits(output,label, reduction = 'mean') 
        break
    torch.cuda.empty_cache()
    # step 2 CG  y
    # step 2.1 init CG params
    def myflatten(x):
        return torch.cat([param.reshape(-1) for param in x], dim = 0)

    # init b
    b = myflatten(torch.autograd.grad(test_loss, model.parameters())).clone().detach().requires_grad_(False) # Ax=b
    torch.cuda.empty_cache()
    grad_trn = myflatten(torch.autograd.grad(train_loss, model.parameters(), create_graph=True, retain_graph=True))
    # init x0
    torch.cuda.empty_cache()
    x = nn.Parameter(b.clone().detach(), requires_grad=False)
    torch.cuda.empty_cache()
    damp = config.damp
    def hvp(g, t, params):
        return myflatten(torch.autograd.grad(g, params, t, retain_graph=True))
    def hvp_with_damp(g, t, params):
        return myflatten(torch.autograd.grad(g, params, t, retain_graph=True)) + damp * t.reshape(-1)
    # = hvp() + damp * t
    r = b.detach() - hvp_with_damp(grad_trn, x, model.parameters()).detach()
    torch.cuda.empty_cache()
    p = r.clone().detach()
    torch.cuda.empty_cache()
    # step 2.2 iterate
    phi_list = []
    i = 0
    while True:
        torch.cuda.empty_cache()

        print(f'current r2:{torch.dot(r, r).item()}, x_mean:{x.mean().item()}'  )
        torch.cuda.empty_cache()
        # stop condition
        with torch.no_grad():
            if (torch.dot(r, r).item() <= 1e-10):
                if config.data_name == 'ml-32m' and config.rec_model == 'lightgcn':
                    print('r2<1e-10, stopping')
                    break
                elif (i >=2):
                    print('r2<1e-10, stopping')
                    break
            elif (torch.dot(r, r).item() <= 1e-6) and (config.rec_model == 'NCF'):
                print('r2<1e-7, NCF, stopping')
                break
            elif (config.rec_model == 'NCF') and (i>=500):
                print('iteraion 500, NCF, stopping')
                break
            
        # iterate
        with torch.no_grad():
            alpha = torch.mul(r,r).sum() / torch.mul(p, hvp_with_damp(grad_trn, p, model.parameters())).sum() # alpha_k
            torch.cuda.empty_cache()
            r_next = r - alpha * hvp_with_damp(grad_trn, p, model.parameters()) # r_k+1
            torch.cuda.empty_cache()
            x = x +  alpha * p # x_k+1
            torch.cuda.empty_cache()
            beta = torch.dot(r_next,r_next) / torch.dot(r,r) # beta_k
            torch.cuda.empty_cache()
            p = r_next + beta * p 
            torch.cuda.empty_cache()
            r = r_next.clone().detach()
            torch.cuda.empty_cache()
        # torch.cuda.empty_cache()
        print(f'alpha{alpha.item()}, x_m{x.mean().item()},r_m{r.mean().item()}, p_m{p.mean().item()},beta{beta.item()}')
        print('------------------------')
        i += 1

    # torch.cuda.empty_cache()
    influence_score_dict = {}
    for synthetic_idx, synthetic_data in tqdm(enumerate(synthetic_data_loader)):
        synthetic_data = synthetic_data.cuda()
        output_syn = model(synthetic_data[:,0].long(), synthetic_data[:,1].long())
        label_syn = synthetic_data[:,2].float() # 1
        loss_syn = torch.nn.functional.binary_cross_entropy_with_logits(output_syn.squeeze(),
                                        label_syn.squeeze(), reduction = 'sum') # 1
        grad_syn = myflatten(torch.autograd.grad(loss_syn, model.parameters()))
        with torch.no_grad():
            influence_score = - (torch.sum(torch.mul(x.reshape(-1), grad_syn.reshape(-1)))).squeeze().squeeze().item()
        influence_score_dict[synthetic_idx] = influence_score
        if synthetic_idx == 0:
    influence_save_name = config.influence_save_name + '_iter_' + config.iter_num + '.pt'
    torch.save(influence_score_dict, os.path.join(influence_save_dir, influence_save_name) )
    return

if __name__=='__main__':
    log_file = None
    evaluate_only = config.evaluate_only
    if not evaluate_only:
        ray_config = {
                'lr':1e-3,
                'embedding_size':128,
                'wd': 1e-5,
                'epoch':config.epoch,
                'eval_epoch':1,
                'patience':150,
                'batch_size':8192
            }
        run_a_trail(ray_config,load_dir=load_dir)



        if log_file is not None:
            log_file.close()




