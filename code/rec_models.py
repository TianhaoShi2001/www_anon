import numpy as np
import torch
import torch.nn as nn
# import torch.functional as F
import torch.nn.functional as F
import os
import time
import pandas as pd
import scipy.sparse as sp

class Personlized_Prompt(nn.Module):
    def __init__(self, config, **kwargs) -> None:
        super().__init__()
        self.config = config
        self.user_num = config.user_num
        self.item_num = config.item_num
        self.padding_index=0
        # self.half()
    def computer(self): # does not need to compute user reprensentation, directly taking the embedding as user/item representations
        return None, None
    def user_encoder(self,users, all_users=None):
        return F.one_hot(users, num_classes = self.item_num+self.user_num).float()
    def item_encoder(self,items, all_items=None):
        return F.one_hot(items + self.user_num, num_classes = self.item_num+self.user_num).float()

class Soft_Prompt(nn.Module):
    def __init__(self, config, **kwargs) -> None:
        super().__init__()
        self.config = config
        self.padding_index=0
        # self.half()
    def computer(self): # does not need to compute user reprensentation, directly taking the embedding as user/item representations
        return None, None
    def user_encoder(self,users, all_users=None):
        u_ = torch.zeros_like(users).to(users.device)
        return F.one_hot(u_, num_classes = 2).float()
    def item_encoder(self,items, all_items=None):
        i_ = torch.ones_like(items).to(items.device)
        return F.one_hot(i_, num_classes = 2).float()








"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Define models here
"""

class LightGCN(nn.Module):
    def __init__(self, config):
        super(LightGCN, self).__init__()
        self.config = config
        self.padding_index = 0
        # self.dataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.num_users  = self.config.user_num
        self.num_items  = self.config.item_num
        self.latent_dim = self.config.embed_size #['latent_dim_rec']
        self.n_layers = self.config.gcn_layers #['lightGCN_n_layers']
        self.keep_prob = self.config.keep_prob #['keep_prob']
        self.A_split = self.config.A_split #['A_split']
        self.dropout_flag = self.config.dropout
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        if self.config.pretrain == 0:
            # nn.init.xavier_uniform_(self.embedding_user.weight, gain=nn.init.calculate_gain('sigmoid'))
            # nn.init.xavier_uniform_(self.embedding_item.weight, gain=nn.init.calculate_gain('sigmoid'))
            # print('use xavier initilizer')
# random normal init seems to be a better choice when lightGCN actually don't use any non-linear activation function
            nn.init.normal_(self.embedding_user.weight, std=self.config.init_emb)
            nn.init.normal_(self.embedding_item.weight, std=self.config.init_emb)
            print('use NORMAL distribution initilizer')
        else:
            self.embedding_user.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
            print('use pretarined data')
        self.f = nn.Sigmoid()
        # self.Graph = self.dataset.Graph
        print(f"lgn is already to go(dropout:{self.config.dropout})")
    
    def _set_graph(self,graph):
        self.Graph = graph.to(self.embedding_user.weight.device)
        self.Graph = self.Graph.to_sparse_csr() # necssary.... for half
        print("Graph Device:", self.Graph.device)

        # print("save_txt")
    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g
    
    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph
    
    def computer(self):
        """
        propagate methods for lightGCN
        """       
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        self.Graph = self.Graph.to(users_emb.device)
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        if self.dropout_flag:
            if self.training:
                print("droping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph        
        else:
            g_droped = self.Graph    
        
        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        #print(embs.size())
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items
    
    def user_encoder(self, users, all_users=None):
        if all_users is None:
            all_users, all_items = self.computer()
        return all_users[users]
    
    def item_encoder(self, items, all_items=None):
        if all_items is None:
            all_users, all_items = self.computer()
        return all_items[items]
    


        
    def F_computer(self,users_emb,items_emb,adj_graph):
        """
        propagate methods for lightGCN
        """       
        # users_emb = self.embedding_user.weight
        # items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        if self.dropout_flag:
            if self.training:
                print("droping")
                raise NotImplementedError("dropout methods are not implemented")
                # g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = adj_graph        
        else:
            g_droped = adj_graph    
        
        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        #print(embs.size())
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items



    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating
    
    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego
    

    def getEmbedding_v2(self, users, items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        # neg_emb = all_items[neg_items]
        # users_emb_ego = self.embedding_user(users)
        # items_emb_ego = self.embedding_item(items)
        # neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, items_emb
    
    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb, 
        userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
                         posEmb0.norm(2).pow(2)  +
                         negEmb0.norm(2).pow(2))/float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        
        return loss, reg_loss
    
    def compute_bce_loss(self, users, items, labels):
        (users_emb, items_emb) = self.getEmbedding_v2(users.long(), items.long())
        matching = torch.mul(users_emb,items_emb)
        scores =  torch.sum(matching,dim=-1)
        bce_loss = F.binary_cross_entropy_with_logits(scores, labels, reduction='mean')
        return bce_loss
       
    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        # print('forward')
        #all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma = torch.sum(inner_pro, dim=1)
        return gamma
    
    def predict(self,users,items):
        users  = torch.from_numpy(users).long().cuda()
        items = torch.from_numpy(items).long().cuda()
        with torch.no_grad():
            all_user_emb, all_item_emb = self.computer()
            users_emb = all_user_emb[users]
            items_emb = all_item_emb[items]
            inner_pro = torch.mul(users_emb,items_emb).sum(dim=-1)
            scores = torch.sigmoid(inner_pro)
        return scores.cpu().numpy()
    

    def predict_changed_graph(self,users,items,changed_graph):
        users  = torch.from_numpy(users).long().cuda()
        items = torch.from_numpy(items).long().cuda()
        with torch.no_grad():
            all_user_emb, all_item_emb = self.F_computer(self.embedding_user.weight,self.embedding_item.weight,changed_graph)
            users_emb = all_user_emb[users]
            items_emb = all_item_emb[items]
            inner_pro = torch.mul(users_emb,items_emb).sum(dim=-1)
            scores = torch.sigmoid(inner_pro)
        return scores.cpu().numpy()





from torch.utils.data import Dataset
class BasicDataset(Dataset):
    def __init__(self):
        print("init dataset")
    

    @property
    def trainDataSize(self):
        raise NotImplementedError
    
    @property
    def testDict(self):
        raise NotImplementedError
    
    @property
    def allPos(self):
        raise NotImplementedError
    
    def getUserItemFeedback(self, users, items):
        raise NotImplementedError
    
    def getUserPosItems(self, users):
        raise NotImplementedError
    
    def getUserNegItems(self, users):
        """
        not necessary for large dataset
        it's stupid to return all neg items in super large dataset
        """
        raise NotImplementedError
    
    def getSparseGraph(self):
        """
        build a graph in torch.sparse.IntTensor.
        Details in NGCF's matrix form
        A = 
            |I,   R|
            |R^T, I|
        """
        raise NotImplementedError
    
class GnnDataset(BasicDataset):
    """
    Dataset type for pytorch \n
    Incldue graph information
    gowalla dataset
    """
    def __init__(self,config,  train, valid, test, path = None, m_users = None, n_item = None):
        # train or test
        # cprint(f'loading [{path}]')
        self.split = config.A_split
        self.folds = config.A_n_fold
        self.mode_dict = {'train': 0, "test": 1}
        self.mode = self.mode_dict['train']
        self.path = path
        self.traindataSize = 0
        self.testDataSize = 0
        self.train = train[['uid','iid','label']]
        self.valid = valid[['uid','iid','label']]
        self.test = test[['uid','iid','label']]

        # self.train = pd.read_csv(train_file)[['uid','iid','lables']]
        # self.valid = pd.read_csv(valid_file)[['uid','iid','lables']]
        # self.test = pd.read_csv(test_file)[['uid','iid','lables']]
        
        self.m_users = m_users if m_users is not None else 1 + max([self.train['uid'].max(),self.valid['uid'].max(),self.test['uid'].max()])
        self.n_items = n_item if n_item is not None else 1 + max([self.train['iid'].max(),self.valid['iid'].max(),self.test['iid'].max()] )
        
        self.testDataSize = self.test.shape[0]
        self.validDataSize = self.valid.shape[0]
        self.train_size = self.train.shape[0]

        self.Graph = None
        print(f"{self.train_size} interactions for normal training")
        print(f"{self.validDataSize} interactions for validation")
        print(f"{self.testDataSize} interactions for testing")
        print(f"{self.m_users} users, {self.n_items} items")
        print(f"{config.dataset} Sparsity : {(self.validDataSize + self.testDataSize+self.train_size) / self.m_users / self.n_items}")
        self._register_graph()
        
        print(":%s is ready to go"%(config.dataset))
    
    def _register_graph(self):
        self.getSparseGraph_mode_a2("graph")
        

 
    @property
    def trainDataSize(self):
        return self.traindataSize
    
    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos

    def _split_A_hat(self,A):
        A_fold = []
        fold_len = (self.m_users + self.n_items) // self.folds
        for i_fold in range(self.folds):
            start = i_fold*fold_len
            if i_fold == self.folds - 1:
                end = self.m_users + self.n_items
            else:
                end = (i_fold + 1) * fold_len
            A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce().cuda())
        return A_fold

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse_coo_tensor(index,data,torch.Size(coo.shape))
        
    def getSparseGraph_mode_a2(self,mode):
        pos_train = self.train[self.train['label']>0].values.copy()
        pos_train[:,1] += self.m_users
        self.trainUser  = self.train['uid'].values.squeeze()
        self.trainItem = self.train['iid']
        print("loading adjacency matrix")
        if self.Graph is None:
            try:
                pre_adj_mat = sp.load_npz(self.path + '/s_pre_adj_mat_'+mode+'.npz')
                print("successfully loaded...")
                norm_adj = pre_adj_mat
            except :
                print("generating adjacency matrix")
                s = time.time()
                pos_train_t = pos_train.copy()
                pos_train_t[:,0] = pos_train[:,1]
                pos_train_t[:,1] = pos_train[:,0]
                pos = np.concatenate([pos_train,pos_train_t],axis=0)
                # negative_indices = pos[(pos[:, 0] < 0) | (pos[:, 1] < 0)]
                # print("Negative indices found in pos array:", negative_indices)
                adj_mat = sp.csr_matrix((pos[:,2], (pos[:,0],pos[:,1])), shape=(self.m_users+self.n_items, self.m_users+self.n_items))
                adj_mat = adj_mat.todok()
                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)
                
                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                end = time.time()
                print(f"costing {end-s}s, saved norm_mat...")
                # sp.save_npz(self.path + '/s_pre_adj_mat_'+mode+'.npz', norm_adj)

            if self.split == True:
                self.Graph = self._split_A_hat(norm_adj)
                print("done split matrix")
            else:
                self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
                self.Graph = self.Graph.coalesce().cuda()
                print("don't split the matrix")
        return self.Graph




    def __build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    def getUserItemFeedback(self, users, items):
        """
        users:
            shape [-1]
        items:
            shape [-1]
        return:
            feedback [-1]
        """
        # print(self.UserItemNet[users, items])
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1,))

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems
    

    def generate_train_dataloader(self,batch_size=1024):
        '''
        generate minibatch data for full training and retrianing
        '''
        data = torch.from_numpy(self.train[['uid','iid','lables']].values)
        train_loader = torch.utils.data.DataLoader(data,shuffle=True,batch_size=batch_size,drop_last=False,num_workers=0)
        return train_loader



class NCF(nn.Module):
    def __init__(self, config, *args, **kwargs) -> None:
        super().__init__()
        self.config = config
        self.padding_index = 0
        self.user_embedding = nn.Embedding(config.user_num, config.embedding_size, padding_idx=self.padding_index)
        self.item_embedding = nn.Embedding(config.item_num, config.embedding_size, padding_idx=self.padding_index)
        
        # MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(config.embedding_size * 2, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, 1) # Output a single score
        )
        nn.init.normal_(self.user_embedding.weight, std=self.config.init_emb)
        nn.init.normal_(self.item_embedding.weight, std=self.config.init_emb)
        print("Created NCF model, user num:", config.user_num, "item num:", config.item_num)

    def user_encoder(self, users, all_users=None):
        return self.user_embedding(users)

    def item_encoder(self, items, all_items=None):
        return self.item_embedding(items)
    
    def computer(self):
        return None, None
    
    def forward(self, users, items):
        user_embedding = self.user_embedding(users)
        item_embedding = self.item_embedding(items)
        concat_embedding = torch.cat([user_embedding, item_embedding], dim=-1)
        output = self.mlp(concat_embedding).squeeze(-1)
        return output

class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs
