import os
from torch.utils.data.dataloader import DataLoader
import pandas as pd
import numpy as np
import torch.optim
from sklearn.metrics import roc_auc_score, log_loss
import torch.nn as nn
import omegaconf
import random 
import time
import numpy as np
import functools
from ray import train, tune
from rec_models import *
from config import config
import json

data_dir = config.data_dir
exp_name = config.exp_name
global_save = True if config.global_save == 1 else False

# current_dir = os.path.abspath(os.path.dirname(__file__))
absolute_path = os.path.join(config.workspace_dir, 'ray_results')
save_dir = os.path.join(config.workspace_dir, f"pretrained_models/{config.data_name}/{config.rec_model}/{exp_name}/{config.trial_name}") 
global_save_dir = os.path.join(config.workspace_dir,f"pretrained_models/{config.data_name}/{config.rec_model}/{exp_name}")
ray_local_dir = absolute_path


def create_folders(filename):
    if filename == None:
        return
    # file_abs_path 
    parent_folder = os.path.abspath(filename)
    # parent_folder = 
    # os.path.dirname(file_abs_path)
    while not os.path.exists(parent_folder):
        os.makedirs(parent_folder)
        print(f"Created parent folder '{parent_folder}'")
        parent_folder = os.path.dirname(parent_folder)

if config.tune_item_only and (not config.evaluate_only):
    load_dir = os.path.join(config.workspace_dir,f"pretrained_models/{config.data_name}/{config.rec_model}/{config.load_model_exp_name}")
else:
    load_dir = global_save_dir

if not config.evaluate_only:
    create_folders(global_save_dir)
    create_folders(save_dir)



def uAUC_me(user, predict, label):
    if not isinstance(predict,np.ndarray):
        predict = np.array(predict)
    if not isinstance(label,np.ndarray):
        label = np.array(label)
    predict = predict.squeeze()
    label = label.squeeze()

    start_time = time.time()
    u, inverse, counts = np.unique(user,return_inverse=True,return_counts=True) # sort in increasing
    index = np.argsort(inverse)
    candidates_dict = {}
    k = 0
    total_num = 0
    only_one_interaction = 0
    computed_u = []
    for u_i in u:
        start_id,end_id = total_num, total_num+counts[k]
        u_i_counts = counts[k]
        index_ui = index[start_id:end_id]
        if u_i_counts ==1:
            only_one_interaction += 1
            total_num += counts[k]
            k += 1
            continue
        # print(index_ui, predict.shape)
        candidates_dict[u_i] = [predict[index_ui], label[index_ui]]
        total_num += counts[k]
        
        k+=1
    print("only one interaction users:",only_one_interaction)
    auc=[]
    only_one_class = 0

    for ui,pre_and_true in candidates_dict.items():
        pre_i,label_i = pre_and_true
        try:
            ui_auc = roc_auc_score(label_i,pre_i)
            auc.append(ui_auc)
            computed_u.append(ui)
        except:
            only_one_class += 1
            # print("only one class")
        
    auc_for_user = np.array(auc)
    print("computed user:", auc_for_user.shape[0], "can not users:", only_one_class)
    uauc = auc_for_user.mean()
    print("uauc for validation Cost:", time.time()-start_time,'uauc:', uauc)
    return uauc, computed_u, auc_for_user

class early_stoper(object):
    def __init__(self,ref_metric='valid_auc', incerase =True,patience=20) -> None:
        self.ref_metric = ref_metric
        self.best_metric = None
        self.increase = incerase
        self.reach_count = 0
        self.patience= patience
        self.patience_add_mark = 0
        # self.metrics = None
    
    def _registry(self,metrics):
        self.best_metric = metrics

    def update(self, metrics):
        if self.best_metric is None:
            self._registry(metrics)
            return True
        else:
            if self.increase and metrics[self.ref_metric] > self.best_metric[self.ref_metric]:
                self.best_metric = metrics
                self.reach_count = 0
                return True
            elif not self.increase and metrics[self.ref_metric] < self.best_metric[self.ref_metric]:
                self.best_metric = metrics
                self.reach_count = 0
                return True 
            else:
                self.reach_count += 1
                return False

    def is_stop(self):
        if self.reach_count>=self.patience:
            if self.best_metric[self.ref_metric] <=0.55: # auc
                if self.patience_add_mark == 0:
                    self.patience += 15
                    self.patience_add_mark = 1
                    return False
                else:
                    return True
            else:
                return True
        else:
            return False

# set random seed   
def run_a_trail(train_config,log_file=None, save_mode=False,save_dir=None,evaluate_only=False,warm_or_cold=None,
                 load_dir = None, save_test_path = None):
    seed=config.run_seed
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

    if config.report_all:
        valid_0 = os.path.join(data_dir,f'{config.data_name}/valid{str(0)}.pkl')
        valid_0 = pd.read_pickle(valid_0)[['uid','iid','label']]
        valid_1 = os.path.join(data_dir,f'{config.data_name}/valid{str(1)}.pkl')
        valid_1 = pd.read_pickle(valid_1)[['uid','iid','label']]
        valid_2 = os.path.join(data_dir,f'{config.data_name}/valid{str(2)}.pkl')
        valid_2 = pd.read_pickle(valid_2)[['uid','iid','label']]
        valid_3 = os.path.join(data_dir,f'{config.data_name}/valid{str(3)}.pkl')
        valid_3 = pd.read_pickle(valid_3)[['uid','iid','label']]

        test_0 = os.path.join(data_dir,f'{config.data_name}/test{str(0)}.pkl')
        test_0 = pd.read_pickle(test_0)[['uid','iid','label']]
        test_1 = os.path.join(data_dir,f'{config.data_name}/test{str(1)}.pkl')
        test_1 = pd.read_pickle(test_1)[['uid','iid','label']]
        test_2 = os.path.join(data_dir,f'{config.data_name}/test{str(2)}.pkl')
        test_2 = pd.read_pickle(test_2)[['uid','iid','label']]
        test_3 = os.path.join(data_dir,f'{config.data_name}/test{str(3)}.pkl')
        test_3 = pd.read_pickle(test_3)[['uid','iid','label']]

    delete_ratio = 0
    if  (config.synthetic_data_path is not None):
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
            #################### sample synthetic data according to the influence score
            if config.influence_save_name:
                influence_save_dir = os.path.join(config.workspace_dir,f"influence_function/{config.data_name}/{config.rec_model}/")
                # load influence
                influence_iters = []
                for iter_n in range(int(config.iter_num) + 1):
                    influence_save_name = config.influence_save_name + '_iter_' + str(iter_n) + '.pt'
                    influence_save_path = os.path.join(influence_save_dir, influence_save_name)
                    if os.path.exists(influence_save_path):
                        influence_dict = torch.load(influence_save_path)
                        assert len(influence_dict) == synthetic_data.shape[0] , "influence dict should have same"
                        # if_list = list(influence_dict.values())
                        influence_iters.append(np.array(list(influence_dict.values())))
                    else:
                        raise Exception(f'influence save path:{influence_save_path} does not exist')
                # process if, d 
                def update_signs(a, a_sign, b_sign, x):
                    a_sign_true = np.where(a_sign < 0, -1, 1)
                    b_sign_true = np.where(b_sign < 0, -1, 1)
                    diff_indices = np.where(a_sign_true != b_sign_true)[0]
                    a_diff = a[diff_indices]
                    top_a_index = np.argsort(np.abs(a_diff))[-x:]
                    b_sign[diff_indices[top_a_index]] = a_sign[diff_indices[top_a_index]]
                    return b_sign
                iid_if_data = pd.DataFrame([])
                iid_if_data['iid'] = synthetic_data['iid'].copy()        
                for iter, influence_iter in enumerate(influence_iters):
                    if iter == 0:
                        if_list = influence_iter
                        if config.data_name == 'ml-32m':
                            if_last_sign = np.ones_like(if_list) * (-1)
                            if_current_sign = np.where(if_list < 0, -1, if_list)
                            if_perturb_sign = update_signs(if_list, if_current_sign,
                                                     if_last_sign, max(1,int(config.perturb_ratio*if_list.shape[0])))
                            iid_if_data['if_score'] = if_perturb_sign
                        else:
                            if_perturb_sign = np.where(if_list < 0, -1, if_list)
                            iid_if_data['if_score'] = if_perturb_sign
                    else:
                        if_last_sign = np.copy(if_perturb_sign)
                        if_list += influence_iter  * (config.ensemble_k * iter + 1 )  # if_current
                        if_current_sign = np.where(if_list <0, -1, if_list)
                        if_perturb_sign = update_signs(if_list, if_current_sign,
                                                     if_last_sign, max(1,int(config.perturb_ratio*if_list.shape[0])))
                        iid_if_data['if_score'] = if_perturb_sign

                if_perturb_sign = list(if_perturb_sign)
                synthetic_data['if_score'] = if_perturb_sign
                if config.if_sampling is not None:  
                    if  'deter' in config.if_sampling:
                        synthetic_data['if_score'] =\
                            synthetic_data.groupby('iid')['if_score'].transform(lambda s: s.where(s.rank(ascending=True) > config.min_item_num, -10000)) 
                        if 'sample_ratio' in train_config.keys():
                            quantile_sample = (synthetic_data['if_score']).quantile(float(train_config['sample_ratio'])) 
                        else:
                            quantile_sample = 0
                        synthetic_data = synthetic_data[synthetic_data['if_score'] <= quantile_sample]
                    
                    synthetic_data = synthetic_data[['uid','iid','label']]
                    print(f'synthetic data have {(len(if_list))} in total, and have {synthetic_data.shape[0]} after sampling')
                    delete_ratio = (len(if_list) - synthetic_data.shape[0]) / len(if_list)
            
                else:
                    raise Exception('do not sampling synthetic data, because if_sampling is None')
            else:
                raise Exception('do not load influence function, because influence_save_name is not provided')
        else:
            raise Exception('synthetic data do not exist')
    else:
        raise Exception('synthetic data is None')
    train_data = pd.concat([train_data,synthetic_data])
            
    train_data_loader = DataLoader(train_data.values, batch_size = train_config['batch_size'], shuffle=True)
    user_num = config.user_num
    item_num = config.item_num
    print("user nums:", user_num, "item nums:", item_num)

    if config.rec_model == 'lightgcn':
        print(train_config['embedding_size'])
        lgcn_config={
            "user_num": int(user_num),
            "item_num": int(item_num),
            "embedding_size": int(train_config['embedding_size']),
            "embed_size": int(train_config['embedding_size']),
            "dataset": config.data_name, # 'ml-1m', #'yahoo-s622-01' #'yahoo-small2' #'yahooR3-iid-001'
            "layer_size": '[64,64]',
            # lightgcn hyper-parameters
            "gcn_layers": train_config['gcn_layers'],
            "keep_prob" : 0.6,
            "A_n_fold": 100,
            "A_split": False,
            "dropout": False,
            "pretrain": 0,
            "init_emb": train_config['init_emb'],
            }
        lgcn_config = omegaconf.OmegaConf.create(lgcn_config)
        gnndata = GnnDataset(lgcn_config, train = train_data, valid=valid_data,test=test_data, 
                             path = data_dir,m_users=user_num,n_item=item_num)
        model = LightGCN(lgcn_config).cuda()
        model._set_graph(gnndata.Graph)
    elif config.rec_model == 'NCF':
        print(train_config['embedding_size'])
        ncf_config={
            "user_num": int(user_num),
            "item_num": int(item_num),
            "embedding_size": int(train_config['embedding_size']),
            "dataset": config.data_name, 
            "hidden_size": train_config['hidden_size'],
            "init_emb": 1e-3,
            }
        ncf_config = omegaconf.OmegaConf.create(ncf_config)
        model = NCF(ncf_config).cuda()
    opt = torch.optim.Adam(model.parameters(),lr=train_config['lr'],weight_decay=train_config['wd'])
    early_stop = early_stoper(ref_metric='valid_auc',incerase=True,patience=train_config['patience'])
    criterion = nn.BCEWithLogitsLoss()

    def evaluate(data):
        pre=[]
        label = []
        users = []
        data_loader = DataLoader(data.values, batch_size = train_config['batch_size'], shuffle=False)
        for batch_id,batch_data in enumerate(data_loader):
            batch_data = batch_data.cuda()
            ui_matching = model(batch_data[:,0].long(),batch_data[:,1].long())
            pre.extend(torch.sigmoid(ui_matching).detach().cpu().numpy())
            label.extend(batch_data[:,-1].cpu().numpy())
            users.extend(batch_data[:,0].cpu().numpy())
        try:
            logloss = log_loss(label,pre)
        except:
            logloss = float('nan')
        try:
            auc = roc_auc_score(label,pre)
        except:
            auc = float('nan')
        try:
            uauc, _, _ = uAUC_me(users, pre, label)
        except:
            uauc = float('nan')
        return auc, uauc, logloss
    
    if evaluate_only:
        model_path = os.path.join(load_dir, 'best_model.pt')
        model.load_state_dict(torch.load(model_path))
        config_path = os.path.join(load_dir, 'best_config.pt')
        best_config = torch.load(config_path)
        model.eval()

        valid_auc, valid_uauc, valid_logloss = evaluate(valid_data)
        test_auc, test_uauc, test_logloss = evaluate(test_data)

        if save_test_path is not None:
            if not os.path.exists(os.path.dirname(save_test_path)):
                os.makedirs(os.path.dirname(save_test_path))
            with open(save_test_path, 'w') as json_file:
                results_dict = {
                    'test_auc':test_auc,
                    'test_uauc':test_uauc,
                    'test_logloss':test_logloss,
                }
                for key, value in best_config.items():
                    results_dict[key] = value
                json_str = json.dumps(results_dict, indent=2)
                json_file.write(json_str)
        return 
    else:
        pass
    
    if config.tune_item_only:
        model_path = os.path.join(load_dir, 'best_model.pt')
        model.load_state_dict(torch.load(model_path))

    valid_auc_top = 0
 
    for epoch in range(train_config['epoch']):
        model.train()
        for bacth_id, batch_data in enumerate(train_data_loader):
            batch_data = batch_data.cuda()
            ui_matching = model(batch_data[:,0].long(),batch_data[:,1].long())
            loss = criterion(ui_matching,batch_data[:,-1].float())
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        if epoch% train_config['eval_epoch']==0:
            model.eval()
            valid_auc,valid_uauc,valid_logloss = evaluate(valid_data)
            test_auc, test_uauc, test_logloss = evaluate(test_data)
            if valid_auc > valid_auc_top:
                valid_auc_top = valid_auc
            updated = early_stop.update({'valid_auc_top':valid_auc_top,'valid_auc':valid_auc, 'valid_uauc':valid_uauc,'test_auc':test_auc, 'test_uauc':test_uauc, 'epoch':epoch})
            
            if not config.report_all:
                report_dict = {'valid_auc_top':valid_auc_top,'valid_auc':valid_auc, 'valid_uauc':valid_uauc,'test_auc':test_auc, 'test_uauc':test_uauc,'delete_ratio':delete_ratio  }
            else:
                val_0_auc,valid_0_uauc, valid_0_logloss = evaluate(valid_0)
                val_1_auc,valid_1_uauc, valid_1_logloss = evaluate(valid_1)
                val_2_auc,valid_2_uauc, valid_2_logloss = evaluate(valid_2)  
                val_3_auc,valid_3_uauc, valid_3_logloss = evaluate(valid_3)   
                test_0_auc,test_0_uauc, test_0_logloss = evaluate(test_0)
                test_1_auc,test_1_uauc, test_1_logloss = evaluate(test_1)
                test_2_auc,test_2_uauc, test_2_logloss = evaluate(test_2)
                test_3_auc,test_3_uauc, test_3_logloss = evaluate(test_3)
                report_dict = {'valid_auc_top':valid_auc_top,'valid_auc':valid_auc, 'test_auc':test_auc, 
                               'val_0_auc':val_0_auc,'valid_0_uauc':valid_0_uauc,'valid_0_logloss': valid_0_logloss,
                               'val_1_auc':val_1_auc,'valid_1_uauc':valid_1_uauc,'valid_1_logloss': valid_1_logloss,                                
                               'val_2_auc':val_2_auc,'valid_2_uauc':valid_2_uauc,'valid_2_logloss': valid_2_logloss,  
                               'val_3_auc':val_3_auc,'valid_3_uauc':valid_3_uauc,'valid_3_logloss': valid_3_logloss,                              
                               'test_0_auc':test_0_auc,'test_0_uauc':test_0_uauc,'test_0_logloss': test_0_logloss, 
                               'test_1_auc':test_1_auc,'test_1_uauc':test_1_uauc,'test_1_logloss': test_1_logloss,
                               'test_2_auc':test_2_auc,'test_2_uauc':test_2_uauc,'test_2_logloss': test_2_logloss,
                               'test_3_auc':test_3_auc,'test_3_uauc':test_3_uauc,'test_3_logloss': test_3_logloss,                               
                                'delete_ratio':delete_ratio 
                                 }
            try:
                train.report(report_dict) 
            except:
                tune.report(**report_dict, training_iteraction = epoch)

            if updated and save_mode:
                save_path = os.path.join(save_dir, 'best_model.pt')
                result_path = os.path.join(save_dir, 'best_result.pt')
                config_path = os.path.join(save_dir, 'best_config.pt')

                if os.path.exists(result_path):
                    best_result = torch.load(result_path)
                else:
                    best_result = valid_auc
                if valid_auc >= best_result:
                    torch.save(valid_auc,result_path)
                    torch.save(model.state_dict(),save_path)
                    torch.save(train_config,config_path)

            print("epoch:{}, valid_auc:{}, test_auc:{}, early_count:{}".format(epoch, valid_auc, test_auc, early_stop.reach_count))
            if early_stop.is_stop():
                print("early stop is reached....!")
                break
    print("train_config:", train_config,"\nbest result:",early_stop.best_metric) 
    if log_file is not None:
        print("train_config:", train_config, "best result:", early_stop.best_metric, file=log_file)
        log_file.flush()
    return



if __name__=='__main__':
    log_file = None
    evaluate_only = config.evaluate_only
    if config.data_name == 'amazon_book':
        if config.rec_model == 'lightgcn':
            ray_config = {
                    'lr':tune.grid_search([1e-2]),
                    'embedding_size':tune.grid_search([128]),
                    'wd': tune.grid_search([1e-7]),
                    'gcn_layers': tune.grid_search([2]),
                    'init_emb':tune.grid_search([1e-3]),
                    'epoch':config.epoch,
                    'eval_epoch':1,
                    'patience':10,
                    'batch_size':8192
                }
        elif config.rec_model == 'NCF':
            ray_config = {
                    'lr':tune.grid_search([1e-3]),
                    'embedding_size':tune.grid_search([128]),
                    'wd': tune.grid_search([1e-7]),
                    'hidden_size':tune.grid_search([64]),
                    'epoch':config.epoch,
                    'eval_epoch':1,
                    'patience':15,
                    'batch_size':8192
                }

    if config.data_name == 'ml-32m':
        if config.rec_model == 'lightgcn':
            ray_config = {
                    'lr':tune.grid_search([1e-2]),
                    'embedding_size':tune.grid_search([128]),
                    'wd': tune.grid_search([1e-7]),
                    'gcn_layers': tune.grid_search([2]),
                    'init_emb':tune.grid_search([1e-3]),
                    'epoch':config.epoch,
                    'eval_epoch':1,
                    'patience':5,
                    'batch_size':tune.grid_search([65536])
                    # 'batch_size':tune.grid_search([65536])
                }
        elif config.rec_model == 'NCF':
            ray_config = {
                    'lr':tune.grid_search([1e-2]),
                    'embedding_size':tune.grid_search([128]),
                    'wd': tune.grid_search([1e-7]),
                    'hidden_size':tune.grid_search([64]),
                    'epoch':config.epoch,
                    'eval_epoch':1,
                    'patience':5,
                    'batch_size':65536
                }
    
    trainable = functools.partial(run_a_trail, log_file=log_file, save_mode=True,save_dir=save_dir,evaluate_only=False,warm_or_cold=None,
                                load_dir = load_dir)
    result = tune.run(
    trainable,
    resources_per_trial={"cpu": 1, "gpu":config.num_per_gpu},
    local_dir = ray_local_dir,
    name = config.data_name + config.rec_model + config.trial_name + exp_name,
    config=ray_config,
    )
    print("Best config is:", result.get_best_config(metric="valid_auc_top", mode="max"))

    if log_file is not None:
        log_file.close()





        
            







