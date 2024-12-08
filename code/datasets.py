from torch.utils.data import Dataset
import os
import pandas as pd
import numpy as np
from config import config
from torch.utils.data.dataloader import default_collate
def convert_title_list_v2(titles):
    titles_ = []
    for x in titles:
        if len(x)>0:
            titles_.append("\""+ x + "\"")
    if len(titles_)>0:
        return ", ".join(titles_)
    else:
        return "unknow"
    
class Mydatasets(Dataset):
    def __init__(self, data, config):
        super().__init__()
        self.config = config
        self.data = data # dataframe
        # uid, iid, title, label
        print("data cols:", self.data.columns)
        print("data size:", self.data.shape)
        self.use_his = False
        if 'his' in self.data.columns:
            self.use_his = True
            try:
                self.data["his"] = self.data["his"].map(list)
                self.data["his_title"] = self.data["his_title"].map(list)
            except:
                pass
            # self.data["his_title"] = self.data["his_title"] #.map(convert_title_list)
        if self.use_his:
            max_length_ = 0
            for x in self.data['his'].values:
                max_length_ = max(max_length_,len(x))
            self.max_length = min(max_length_, 15) # average: only 5 
            print("amazon datasets, max history length:", self.max_length)
        return
    
    def __len__(self):
        return len(self.data)
    
    def collater(self, samples):
        return default_collate(samples)
    
    # todo new get item
    def __getitem__(self, index):
        data_point = self.data.iloc[index]
        if self.use_his:
            a = data_point['his']
            InteractedNum = len(a)
            if a[0] == 0:
                InteractedNum -= 1

            if len(a) < self.max_length:
                b = [0]* (self.max_length-len(a)) # assuming padding idx is zero
                b.extend(a)
            elif len(a)> self.max_length:
                b = a[-self.max_length:]
                InteractedNum = self.max_length
            else:
                b = a
            return {
                "uid": data_point['uid'],
                "his_pad": np.array(b),
                # "his": data_point['his'],
                "his_title": convert_title_list_v2(data_point['his_title'][-InteractedNum:]),
                "iid": data_point["iid"],
                "title": "\""+data_point["title"]+"\"",
                "InteractedNum": InteractedNum,
                "label": data_point['label']
            }
        one_sample = {
            "uid": data_point['uid'],
            "iid": data_point["iid"],
            "title": data_point["title"].strip(' '),
            "label": data_point['label']
        }
        return one_sample 

class cf_datasets(Dataset):
    def __init__(self, data, config):
        super().__init__()
        self.config = config
        self.data = data # dataframe
        # uid, iid, title, label
        print("data cols:", self.data.columns)
        print("data size:", self.data.shape)
        self.use_his = False
        if 'his' in self.data.columns:
            self.use_his = True
            try:
                self.data["his"] = self.data["his"].map(list)
                self.data["his_title"] = self.data["his_title"].map(list)
            except:
                pass
            # self.data["his_title"] = self.data["his_title"] #.map(convert_title_list)
        if self.use_his:
            max_length_ = 0
            for x in self.data['his'].values:
                max_length_ = max(max_length_,len(x))
            self.max_length = min(max_length_, 15) # average: only 5 
            print("amazon datasets, max history length:", self.max_length)
        return
    
    def __len__(self):
        return len(self.data)
    
    def collater(self, samples):
        return default_collate(samples)
    
    # todo new get item
    def __getitem__(self, index):
        data_point = self.data.iloc[index]
        if self.use_his:
            a = data_point['his']
            InteractedNum = len(a)
            if a[0] == 0:
                InteractedNum -= 1

            if len(a) < self.max_length:
                b = [0]* (self.max_length-len(a)) # assuming padding idx is zero
                b.extend(a)
            elif len(a)> self.max_length:
                b = a[-self.max_length:]
                InteractedNum = self.max_length
            else:
                b = a
            return {
                "uid": data_point['uid'],
                "his_pad": np.array(b),
                "iid": data_point["iid"],
                "InteractedNum": InteractedNum,
                "label": data_point['label']
            }
        one_sample = {
            "uid": data_point['uid'],
            "iid": data_point["iid"],
            "label": data_point['label']
        }
        return one_sample 
    
class IFdatasets(Dataset):
    def __init__(self, data, config = None):
        super().__init__()
        self.config = config
        self.data = data # dataframe
        # uid, iid, title, label
        print("data cols:", self.data.columns)
        print("data size:", self.data.shape)
        self.use_his = False
        if 'his' in self.data.columns:
            self.use_his = True
            # self.data = self.data[['uid','iid','title','his', 'his_title','label','timestamp']]
            self.data["his"] = self.data["his"].map(list)
            self.data["his_title"] = self.data["his_title"].map(list)
            self.data["his_title"] = self.data["his_title"] #.map(convert_title_list)
        if self.use_his:
            max_length_ = 0
            for x in self.data['his'].values:
                max_length_ = max(max_length_,len(x))
            self.max_length = min(max_length_, 15) # average: only 5 
            print("amazon datasets, max history length:", self.max_length)
        return
    
    def __len__(self):
        return len(self.data)
    
    def collater(self, samples):
        return default_collate(samples)
    
    # todo new get item
    def __getitem__(self, index):
        data_point = self.data.iloc[index]
        if self.use_his:
            a = data_point['his']
            InteractedNum = len(a)
            if a[0] == 0:
                InteractedNum -= 1

            if len(a) < self.max_length:
                b = [0]* (self.max_length-len(a)) # assuming padding idx is zero
                b.extend(a)
            elif len(a)> self.max_length:
                b = a[-self.max_length:]
                InteractedNum = self.max_length
            else:
                b = a
            return {
                "uid": data_point['uid'],
                "his_pad": np.array(b),
                # "his": data_point['his'],
                "his_title": convert_title_list_v2(data_point['his_title'][-InteractedNum:]),
                "iid": data_point["iid"],
                "title": "\""+data_point["title"]+"\"",
                "InteractedNum": InteractedNum,
                "label": 1, # 只是用一下
                "pos_label": 1,
                'neg_label': 0,
                "pos_if": data_point["pos_if"],
                "neg_if": data_point["neg_if"],
                "dpo_weight": data_point["dpo_weight"],
            }
        one_sample = {
            "uid": data_point['uid'],
            "iid": data_point["iid"],
            "title": data_point["title"].strip(' '),
            "pos_label": 1,
            'neg_label': 0,
            "label": 1, 
            "pos_if": data_point["pos_if"],
            "neg_if": data_point["neg_if"],
            "dpo_weight": data_point["dpo_weight"],
        }
        return one_sample 



class Datasets_builder:
    def __init__(self, config):
        self.config = config # generate_synthetic
        dataset_dir = config.dataset_dir


        if self.config.dataset_version == 'v1':
            dataset_cls = Mydatasets # todo  change other get items 
        else:
            raise Exception('Not implemented dataset version')

        self.datasets = {}
        # get train data  
        train_paths = []
        for i in range(self.config.train_start_num, self.config.train_start_num + self.config.train_total_num):
            train_paths.append(os.path.join(dataset_dir, f'train{i}.pkl'))
            if i != self.config.train_start_num + self.config.train_total_num - 1:
                train_paths.append(os.path.join(dataset_dir, f'valid{i}.pkl'))
        train_data = self.read_data(train_paths)
        self.datasets['train'] = dataset_cls(train_data, config) # to do, maybe use config to load side information # config.id2title/id2page ....
        
        # get valid & test data
        valid_path = os.path.join(dataset_dir, f'valid{self.config.valid_num}.pkl')
        valid_data = self.read_data(valid_path)
        self.datasets['valid'] = dataset_cls(valid_data, config)

        test_path = os.path.join(dataset_dir, f'test{self.config.test_num}.pkl')
        test_data = self.read_data(test_path)
        self.datasets['test'] = dataset_cls(test_data, config)

        if self.config.run_dpo and (not self.config.evaluate_only):
            if self.config.calculate_if_data_path is not None:
                if os.path.exists(self.config.calculate_if_data_path):
                    if_test_data = self.read_data(self.config.calculate_if_data_path)
                    self.datasets['if_test'] = dataset_cls(if_test_data, config)
                else:
                    raise Exception('the if data path does not exist')
            else:
                if self.config.cal_influence_on == 'valid':
                    if_test_path = os.path.join(dataset_dir, f'valid{self.config.calculate_if_data_num}.pkl')
                    if_test_data = self.read_data(if_test_path)
                    self.datasets['if_test'] = dataset_cls(if_test_data, config)
                if self.config.cal_influence_on == 'test':
                    if_test_path = os.path.join(dataset_dir, f'test{self.config.calculate_if_data_num}.pkl')
                    if_test_data = self.read_data(if_test_path)
                    self.datasets['if_test'] = dataset_cls(if_test_data, config)
                
            if_train_path = os.path.join(dataset_dir, f'valid{self.config.calculate_if_train_data_num}.pkl')
            if_train_data = self.read_data(if_train_path)
            self.datasets['if_train'] = dataset_cls(if_train_data, config)
            
            if_z_data = self.generate_if_data()
            self.datasets['if_z'] = cf_datasets(if_z_data, config)
        if self.config.train_synthetic:
            synthetic_data_path = self.config.synthetic_data_path
            self.synthetic_dataset_for_training = self.load_synthetic_data_for_training(synthetic_data_path) # 这里不能让它成为synthetic，目的是更改训练
            self.datasets['train'] = pd.concat([self.train_dataset, self.synthetic_dataset_for_training])

        if self.config.generate_synthetic:
            synthetic_data_for_generation = self.generate_synthetic_data()
            self.datasets['synthetic'] = dataset_cls(synthetic_data_for_generation, config)


    def generate_if_data(self):
        dpo_generate_num = self.config.dpo_generate_num
        # most_simple_generate_version 
        if (dpo_generate_num !=0 ) and  (dpo_generate_num is not None):
            dataset_dir = config.dataset_dir
            if_z_test_path = os.path.join(dataset_dir, f'test{self.config.dpo_test_num}.pkl') # z
            if_z_test_data = self.read_data(if_z_test_path)
            test_item = if_z_test_data.drop_duplicates(subset=['iid'])
            test_item_ids = test_item['iid'].to_list()   
            test_item_titles = test_item['title'].to_list()                   

            # 现在选取warm user，且只保留user最后一组prompt
            user_cnt = self.datasets['train'].data['uid'].value_counts()
            warm_users = user_cnt[user_cnt>=10].index.unique().tolist()
            train_data = self.datasets['train'].data # ['uid','iid','title', 'his', 'his_title','label']
            warm_data = train_data[train_data['uid'].isin(warm_users)]
            warm_data.sort_values(by = 'timestamp', ascending=False, inplace=True)
            warm_data.drop_duplicates(subset=['uid'], inplace=True)
            synthetic_interactions = []
            try:
                np.random.seed(self.config.generate_seed) # set random seed
            except:
                np.random.seed(2024)
            for (id, title) in zip(test_item_ids, test_item_titles):
                sampled_data = warm_data.sample(n = int(dpo_generate_num))
                sampled_data['iid'] = id        
                sampled_data['title'] = title  
                synthetic_interactions.append(sampled_data)       
            synthetic_data = pd.concat(synthetic_interactions) # here for testing
        return synthetic_data


    def generate_synthetic_data_v0(self):
        generate_num = self.config.generate_num
        # most_simple_generate_version 
        if (generate_num !=0 ) and  (generate_num is not None):
            test_item = self.datasets['test'].data.drop_duplicates(subset=['iid'])
            test_item_ids = test_item['iid'].to_list()   
            test_item_titles = test_item['title'].to_list()                   

            # 现在选取warm user，且只保留user最后一组prompt
            user_cnt = self.datasets['train'].data['uid'].value_counts()
            warm_users = user_cnt[user_cnt>=10].index.unique().tolist()
            train_data = self.datasets['train'].data # ['uid','iid','title', 'his', 'his_title','label']
            warm_data = train_data[train_data['uid'].isin(warm_users)]
            warm_data.sort_values(by = 'timestamp', ascending=False, inplace=True)
            warm_data.drop_duplicates(subset=['uid'], inplace=True)

            synthetic_interactions = []
            if isinstance(self.config.generate_seed, int):
                generate_seed = self.config.generate_seed
            else:
                generate_seed= 2024
            for (id, title) in zip(test_item_ids, test_item_titles):
                np.random.seed(generate_seed)
                generate_seed += 1
                sampled_data = warm_data.sample(n = int(generate_num))
                sampled_data['iid'] = id        
                sampled_data['title'] = title  
                synthetic_interactions.append(sampled_data)       
            synthetic_data = pd.concat(synthetic_interactions) # here for testing
        return synthetic_data
    
    @property
    def generate_synthetic_data(self):
        if self.config.generate_synthetic_version == 'v0':
            return self.generate_synthetic_data_v0
        elif self.config.generate_synthetic_version == 'v1':
            return self.generate_synthetic_data_v1
        elif self.config.generate_synthetic_version == 'v2':
            return self.generate_synthetic_data_v2
        elif self.config.generate_synthetic_version == 'v3':
            return self.generate_synthetic_data_v3
        elif self.config.generate_synthetic_version == 'v4':
            return self.generate_synthetic_data_v4
        else:
            raise Exception('Generate synthetic data version not implemented')
    
    def generate_synthetic_data_v1(self): 
        generate_num = self.config.generate_num
        # most_simple_generate_version 
        if (generate_num !=0 ) and  (generate_num is not None):
            test_item = self.datasets['test'].data.drop_duplicates(subset=['iid'])
            test_item_ids = test_item['iid'].to_list()   
            test_item_titles = test_item['title'].to_list()                   
            valid_users = self.datasets['valid'].data['uid'].unique().tolist()

            user_cnt = self.datasets['train'].data['uid'].value_counts()
            warm_users = user_cnt[user_cnt>=10].index.unique().tolist()
            train_data = self.datasets['train'].data # ['uid','iid','title', 'his', 'his_title','label']
            warm_data = train_data[train_data['uid'].isin(warm_users)]
            warm_data.sort_values(by = 'timestamp', ascending=False, inplace=True)
            warm_data.drop_duplicates(subset=['uid'], inplace=True)
            warm_data = warm_data[~warm_data['uid'].isin(valid_users)]
            synthetic_interactions = []
            np.random.seed(2024) # set random seed
            sampled_data = warm_data.sample(n = int(generate_num))
            for (id, title) in zip(test_item_ids, test_item_titles):
                # sampled_data = warm_data.sample(n = int(generate_num))
                sampled_data_copy = sampled_data.copy()
                sampled_data_copy['iid'] = id        
                sampled_data_copy['title'] = title  
                synthetic_interactions.append(sampled_data_copy)       
            synthetic_data = pd.concat(synthetic_interactions) # here for testing
        return synthetic_data

    def generate_synthetic_data_v2(self): # version 2,对于每个item，采样相同user，且user一定在valid set里出现过。
        generate_num = self.config.generate_num
        # most_simple_generate_version 
        if (generate_num !=0 ) and  (generate_num is not None):
            test_item = self.datasets['test'].data.drop_duplicates(subset=['iid'])
            test_item_ids = test_item['iid'].to_list()   
            test_item_titles = test_item['title'].to_list()                   

            valid_users = self.datasets['valid'].data['uid'].unique().tolist()

            user_cnt = self.datasets['train'].data['uid'].value_counts()
            warm_users = user_cnt[user_cnt>=10].index.unique().tolist()
            train_data = self.datasets['train'].data # ['uid','iid','title', 'his', 'his_title','label']
            warm_data = train_data[train_data['uid'].isin(warm_users)]
            warm_data.sort_values(by = 'timestamp', ascending=False, inplace=True)
            warm_data.drop_duplicates(subset=['uid'], inplace=True)
            warm_data = warm_data[warm_data['uid'].isin(valid_users)]
            synthetic_interactions = []
            if isinstance(self.config.generate_seed, int):
                generate_seed = self.config.generate_seed
            else:
                generate_seed= 2024
            np.random.seed(generate_seed) # set random seed
            sampled_data = warm_data.sample(n = int(generate_num))
            for (id, title) in zip(test_item_ids, test_item_titles):
                # sampled_data = warm_data.sample(n = int(generate_num))
                sampled_data_copy = sampled_data.copy()
                sampled_data_copy['iid'] = id        
                sampled_data_copy['title'] = title  
                synthetic_interactions.append(sampled_data_copy)       
            synthetic_data = pd.concat(synthetic_interactions) # here for testing
        return synthetic_data

    def generate_synthetic_data_v3(self): 
        generate_num = self.config.generate_num
        # most_simple_generate_version 
        if (generate_num !=0 ) and  (generate_num is not None):
            test_item = self.datasets['test'].data.drop_duplicates(subset=['iid'])
            test_item_ids = test_item['iid'].to_list()   
            test_item_titles = test_item['title'].to_list()                   

            valid_users = self.datasets['valid'].data['uid'].unique().tolist()

            user_cnt = self.datasets['train'].data['uid'].value_counts()
            warm_users = user_cnt[user_cnt>=10].index.unique().tolist()
            train_data = self.datasets['train'].data # ['uid','iid','title', 'his', 'his_title','label']
            warm_data = train_data[train_data['uid'].isin(warm_users)]
            warm_data.sort_values(by = 'timestamp', ascending=False, inplace=True)
            warm_data.drop_duplicates(subset=['uid'], inplace=True)
            warm_data = warm_data[~warm_data['uid'].isin(valid_users)]
            synthetic_interactions = []
            try:
                np.random.seed(self.config.generate_seed) # set random seed
            except:
                np.random.seed(2024)
            # sampled_data = warm_data.sample(n = int(generate_num))
            for (id, title) in zip(test_item_ids, test_item_titles):
                sampled_data_copy = warm_data.sample(n = int(generate_num))
                sampled_data_copy['iid'] = id        
                sampled_data_copy['title'] = title  
                synthetic_interactions.append(sampled_data_copy)       
            synthetic_data = pd.concat(synthetic_interactions) # here for testing
        return synthetic_data

    def generate_synthetic_data_v4(self): 
        generate_num = self.config.generate_num
        # most_simple_generate_version 
        if (generate_num !=0 ) and  (generate_num is not None):
            test_item = self.datasets['test'].data.drop_duplicates(subset=['iid'])
            test_item_ids = test_item['iid'].to_list()   
            test_item_titles = test_item['title'].to_list()                   

            valid_users = self.datasets['valid'].data['uid'].unique().tolist()
            user_cnt = self.datasets['train'].data['uid'].value_counts()
            warm_users = user_cnt[user_cnt>=10].index.unique().tolist()
            train_data = self.datasets['train'].data # ['uid','iid','title', 'his', 'his_title','label']
            warm_data = train_data[train_data['uid'].isin(warm_users)]
            warm_data.sort_values(by = 'timestamp', ascending=False, inplace=True)
            warm_data.drop_duplicates(subset=['uid'], inplace=True)
            warm_data = warm_data[warm_data['uid'].isin(valid_users)]
            synthetic_interactions = []
            if isinstance(self.config.generate_seed, int):
                generate_seed = self.config.generate_seed
            else:
                generate_seed= 2024
            for (id, title) in zip(test_item_ids, test_item_titles):
                np.random.seed(generate_seed)
                generate_seed += 1
                # sampled_data = warm_data.sample(n = int(generate_num))
                sampled_data_copy = warm_data.sample(n = int(generate_num))
                sampled_data_copy['iid'] = id        
                sampled_data_copy['title'] = title  
                synthetic_interactions.append(sampled_data_copy)       
            synthetic_data = pd.concat(synthetic_interactions) # here for testing
        return synthetic_data



    def process_synthetic_data_for_training(self, synthetic_data_path = None):
        if synthetic_data_path is None:
            raise Exception("set train_synthetic=True, while synthetic data is None")
        elif not os.path.exists(synthetic_data_path):
            raise Exception("set train_synthetic=True, while synthetic data path can not be loaded")
        else:
            print('loading synthetic dataset for training...')
            synthetic_data = pd.read_pickle(synthetic_data_path) # here for training
            # convert logits to 0,1 label
            train_data = self.train_dataset
            train_p1 = float(train_data['label'].mean())
            best_thre = 0
            best_diff = 100
            for thre in list(np.linspace(0,5,100)):
                predict_p1 = synthetic_data[synthetic_data['predicted_logit']>=thre].shape[0] / synthetic_data.shape[0]
                if abs(train_p1 - predict_p1) < best_diff:
                    best_diff = abs(train_p1 - predict_p1) 
                    best_thre = thre
            synthetic_data['label'] = np.where(synthetic_data['predicted_logit']>=best_thre,1,0 )
            synthetic_data = synthetic_data[['uid','iid','label','his']]
            id2title_dict = np.load(self.config.id2title_dict_path, allow_pickle=True)
            def convert_id2title(df):
                title = df['iid'].apply(lambda x: id2title_dict[x])
                def delete_0_in_his(his):
                    non_zero_his = list(filter(lambda x: x != 0, his))
                    if non_zero_his == []:
                        non_zero_his = [0]
                    return non_zero_his
                his = df['his'].apply(delete_0_in_his)
                new_df = df.copy()
                new_df['his'] = his
                new_df['title'] = title
                new_df['his_title'] = new_df['his'].apply(lambda x: [id2title_dict[item] for item in x])
                new_df['timestamp'] = pd.to_datetime('2022-10-01')
                new_df['env'] = 0
                return new_df
            synthetic_data_new = convert_id2title(synthetic_data)
        return synthetic_data_new

    def read_data(self,data_paths):
        if isinstance(data_paths, list):
            data_list = []
            for path in data_paths:
                data_list.append(pd.read_pickle(path))
            data = pd.concat(data_list)
            return data
        elif isinstance(data_paths, str):
            if os.path.exists(data_paths):
                data = pd.read_pickle(data_paths)
                return data
            else: 
                raise Exception("can not load data")
        else:
            raise Exception("can not load data")
        
    