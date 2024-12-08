import logging
import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from transformers import LlamaTokenizer
from modeling_llama  import LlamaForCausalLM 
import warnings
from rec_models import *
from peft import *
from utils import *
import random
import torch.utils.checkpoint as checkpoint

class CoLLM(nn.Module):

    def __init__(self, config, low_resource=True, proj_token_num=1,):
        super().__init__()
        self.config = config
        self.low_resource = low_resource
        self.proj_token_num = proj_token_num

        self.init_modules()
        self.init_load_modules()
        self.init_others()
        self.device =  self.config.device 
        self.set_answer_type(config.get('ans_type'))
        

    def init_modules(self):
        print("loading CoLLM ... ")
        print('Loading recommendation model')
        self.rec_model_type = self.config.rec_model
        self.rec_encoder = self.init_rec_encoder(self.config.rec_model, self.config)
        # try:
        self.if_rec_model = None

        print('Loading recommendation model Done')

        print('Loading LLAMA')
        print('llama model path', self.config.llama_model)
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(self.config.llama_model, use_fast=True)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token

        if self.low_resource:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                self.config.llama_model,
                torch_dtype=torch.bfloat16,
                load_in_8bit=True,
                device_map={'': 0}
            )
            try:
                self.llama_model = prepare_model_for_int8_training(self.llama_model)
            except:
                self.llama_model = prepare_model_for_kbit_training(self.llama_model)
        else:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                self.config.llama_model,
                torch_dtype=torch.bfloat16,
            )
        for name, param in self.llama_model.named_parameters():
            param.requires_grad = False
        print('Loading LLAMA Done')
        self.use_lora = False
        if self.config.lora_config is not None and self.config.lora_config.use_lora:
            print("Setting Lora")
            self.use_lora = True
            peft_config = LoraConfig(
                r=self.config.lora_config.r,
                lora_alpha=self.config.lora_config.alpha,
                target_modules=self.config.lora_config.target_modules,
                lora_dropout=self.config.lora_config.dropout,
                bias="none",
                task_type="CAUSAL_LM"
            ) 
            self.llama_model = get_peft_model(self.llama_model, peft_config)
        
        if self.rec_encoder is not None and 'prompt' not in self.config.rec_model:
            print("type:", type(self.config.proj_mid), self.config.proj_mid)
            self.llama_proj = nn.Sequential(
                nn.Linear(self.config.embedding_size, self.config.embedding_size*int(self.config.proj_mid)),  # ml100=>5
                nn.ReLU(),
                nn.Linear(self.config.embedding_size*int(self.config.proj_mid), self.config.llama_hidden_size * self.proj_token_num),
            )
            
        elif self.rec_encoder is not None and self.config.rec_model=="personlized_prompt": #'prompt' in rec_model:
            # identical mapping function, i.e., f(x)=x
            print("personalized prompt learning....")
            self.llama_proj = nn.Linear(self.config.item_num+self.config.user_num, self.config.llama_hidden_size * self.proj_token_num,bias=False) #identical_map()
        elif self.rec_encoder is not None and self.config.rec_model=="soft_prompt": #'prompt' in rec_model:
            # identical mapping function, i.e., f(x)=x
            print("soft prompt learning....")
            self.llama_proj = nn.Linear(2, self.config.llama_hidden_size * self.proj_token_num,bias=False) #identical_map()
        else:
            self.llama_proj = None

    def init_load_modules(self):
        
        lora_ckpt_path = self.config.lora_ckpt
        if lora_ckpt_path:
            if os.path.exists(lora_ckpt_path):
                print("Load MiniGPT4Rec lora Checkpoint: {}".format(lora_ckpt_path))
                lora_ckpt = torch.load(lora_ckpt_path, map_location="cpu")
                set_peft_model_state_dict(self.llama_model, peft_model_state_dict=lora_ckpt['lora'])
            else:
                raise Exception(f'try to load lora checkpoint {lora_ckpt_path}, but file path does not exist')
            

        proj_ckpt_path = self.config.proj_ckpt 
        if proj_ckpt_path:
            if os.path.exists(proj_ckpt_path):
                print("Load MiniGPT4Rec Checkpoint: {}".format(proj_ckpt_path))
                proj_ckpt = torch.load(proj_ckpt_path, map_location="cpu")
                msg = self.load_state_dict(proj_ckpt['proj'], strict=False)
            else:
                raise Exception(f'try to load projection checkpoint {proj_ckpt_path}, but file path does not exist')
        else:
            print('do not load proj parameter because the proj ckpt path is None')

        if self.config.pretrained_path:
            if self.rec_encoder is not None and os.path.exists(self.config.pretrained_path):
                self.rec_encoder.load_state_dict(torch.load(self.config.pretrained_path, map_location="cpu"), strict=False)
                print("successfully load the recommendation model......")
                if self.if_rec_model is not None and os.path.exists(self.config.pretrained_path):
                    self.if_rec_model.load_state_dict(torch.load(self.config.pretrained_path, map_location="cpu"), strict=False)
                    print("successfully load the recommendation model......")
            elif not os.path.exists(self.config.pretrained_path):
                raise Exception('pretrained recommendation model is not None, but cannot be loaded')
            elif self.rec_encoder is None:
                print('do not load recommendation model because the rec encoder is None')
                # raise Exception('rec encoder is None')
        else:
            print("do not load recommendation model because the pretrain path is None")
        
        # freeze or activate model params
        if self.llama_proj is not None:
            if self.config.freeze_proj:
                for name, param in self.llama_proj.named_parameters():
                    param.requires_grad = False
                self.llama_proj = self.llama_proj.eval()
                self.llama_proj.train = disabled_train 
                logging.info("!!!! freeze llama_proj...")
            else:
                for name, param in self.llama_proj.named_parameters():
                    param.requires_grad = True
                self.llama_proj = self.llama_proj.train()
                logging.info("!!!! avtivate llama_proj...")
                print("!!!! avtivate llama_proj...")


        if self.rec_encoder is not None:
            if self.config.freeze_rec:
                for name, param in self.rec_encoder.named_parameters():
                    param.requires_grad = False
                self.rec_encoder = self.rec_encoder.eval()
                self.rec_encoder.train = disabled_train
                logging.info("freeze rec encoder")
                print("freeze rec encoder")
            else:
                for name, param in self.rec_encoder.named_parameters():
                    param.requires_grad = True
                self.rec_encoder =self.rec_encoder.train()
                # model.rec_encoder.train = nn.Module.train
                # model.rec_encoder.train()
                logging.info("activate rec encoder")
                print("activate rec encoder")
        
        if self.llama_model is not None:
            if self.config.freeze_lora:
                print("freeze lora...")
                for name, param in self.llama_model.named_parameters():
                    param.requires_grad = False

    def init_others(self):
        self.max_txt_len = self.config.max_txt_len
        self.end_sym = self.config.end_sym
        self.has_print_prompt=False

        if self.config.prompt_path:
            with open(self.config.prompt_path, 'r') as f:
                raw_prompts = f.read().splitlines()
            filted_prompts = [raw_prompt for raw_prompt in raw_prompts]
            self.prompt_list = [self.config.prompt_template.format(p) for p in filted_prompts]
            print('Load {} training prompts'.format(len(self.prompt_list)))
            print('Prompt List: \n{}'.format(self.prompt_list))
            self.has_pri_decode=False
            self.prompt_list_p = None
        else:
            self.prompt_list = []
            self.prompt_list_p = None

    @classmethod
    def init_rec_encoder(self,rec_model, config):
        if rec_model == "MF":
            print("### rec_encoder:", "MF")
            rec_model = MatrixFactorization(config)
        elif rec_model == "lightgcn":
            print("### rec_encoder:", "lightgcn")
            rec_model = LightGCN(config)
        elif rec_model == "random_mf":
            print("### rec_encoder:", "random_mf")
            rec_model = random_mf(config)
        elif rec_model == 'soft_prompt':
            print("### rec_encoder:", "soft_prompt")
            rec_model = Soft_Prompt(config)
        else:
            rec_model = None
            warnings.warn(" the input rec_model is not MF, LightGCN or sasrec, or DCN, we won't utilize the rec_encoder directly.")
        return rec_model


    def to_be_trained(self):
        if self.use_lora:
            return True
        id_terms = ["<uid>", "<his>", "<iid>", "<DCNFeature>"]
        for prompt in self.prompt_list:
            for id_term in id_terms:
                if id_term in prompt:
                    return True
        return False
    
    def set_mode(self, mode):
        '''
        mode \in ['v1','v2',None]
        '''
        self.run_mode_ = mode
    
    def rec_to_cpu(self):
        self.rec_encoder.to("cpu")
        self.rec_encoder.float()
    
    def set_answer_type(self,ans_type_mode):
        if ans_type_mode == 'v1':
            self.pos_ans = ["former"]
            self.neg_ans = ["latter"]
        elif ans_type_mode == 'v2':
            self.pos_ans = ['Yes']
            self.neg_ans = ['No']
            pos_ans_id = self.llama_tokenizer(self.pos_ans[0],add_special_tokens=False).input_ids[0]
            neg_ans_id = self.llama_tokenizer(self.neg_ans[0],add_special_tokens=False).input_ids[0]
            print("answer token ids: pos:",pos_ans_id, "neg ids:", neg_ans_id)
            
        else:
            raise NotImplementedError("not implement this types of answers")
    def print_prompt(self):
        print('Prompt Pos Example \n{} {} or {}'.format(random.choice(self.prompt_list),self.pos_ans[0],self.neg_ans[0]))


    def encode_recdata_v2(self, sample, ids_order=None):  # used for stage2
        if self.rec_encoder is None:
            return None, None
        device = sample['uid'].device

        with self.maybe_autocast():
            batch_size = sample['uid'].shape[0]
            llama_hidden_size = self.config.llama_hidden_size
            all_user_embeds, all_item_embeds = self.rec_encoder.computer()
            if self.rec_model_type == "sasrec":  # for sasrec, there is no user encoder but just seqs encoder, we take it to get user representation
                user_embeds = self.rec_encoder.seq_encoder(sample['sas_seq']).unsqueeze(-2)
            elif self.rec_model_type == "DCN" or self.rec_model_type == "DIN":
                """
                not really user embeding, but the embedding merged for one sample point
                """
                user_embeds = self.rec_encoder.all_encode(sample['uid'],sample['iid'],sample['sas_seq'][:,-10:]).unsqueeze(-2)
            else:
                user_embeds = self.rec_encoder.user_encoder(sample['uid'], all_users=all_user_embeds).unsqueeze(-2)
            targetItem_embed = self.rec_encoder.item_encoder(sample['iid'], all_items=all_item_embeds).unsqueeze(-2)

            user_embeds_llama = self.llama_proj(user_embeds).reshape(batch_size,-1, self.proj_token_num, llama_hidden_size)
            targetItem_embeds_llama = self.llama_proj(targetItem_embed).reshape(batch_size,-1, self.proj_token_num, llama_hidden_size)
            
            sample_embeds_llama = {
                'User_emb': user_embeds_llama.reshape(batch_size,-1, llama_hidden_size),
                'TargetItem_emb': targetItem_embeds_llama.reshape(batch_size,-1, llama_hidden_size),
                'InteractedItems_embs': None,
                'merged_embs': None,
            }
        sample_atts_llama = None

        return sample_embeds_llama, sample_atts_llama

    def recprompt_wrap_v2(self, samples, ori_samples, atts_sample, prompt): # used for stage 2
        if prompt:
            prompt_ori = prompt
            split_symbol = ["<uid>", "<his>", "<his_title>", "<iid>", "<title>"]
            batch_size = ori_samples['uid'].shape[0]
            bos = "<s>"
            unk_ = self.llama_tokenizer.unk_token #"<unk>"
            unk_ = ".".join([unk_]*self.proj_token_num)
            prompt = bos + prompt # add the bos
            prompt = prompt.replace("<uid>", unk_)
            prompt = prompt.replace("<iid>", unk_)
            prompt = prompt.replace("<DCNFeature>", unk_)
            prompt_list = []
            
            
            for k in range(batch_size):
                prompt_ = prompt+""
                if 'InteractedNum' in ori_samples.keys():
                    prompt_ = prompt_.replace('<his>', ', '.join([unk_]*ori_samples['InteractedNum'][k]))
                    prompt_ = prompt_.replace("<his_title>", ori_samples['his_title'][k]) # todo
                prompt_ = prompt_.replace("<title>", ori_samples['title'][k])
                prompt_list.append(prompt_)
            
            if not self.has_print_prompt:
                print("prompt example:", random.choice(prompt_list))
                self.has_print_prompt = True
            
            self.llama_tokenizer.padding_side = "left"
            prompts_tokens = self.llama_tokenizer(
            prompt_list,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(ori_samples['uid'].device)
            unk_token_id = self.llama_tokenizer.unk_token_id
            if not self.has_pri_decode:
                print("#######prmpt decoded example: ",' '.join(self.llama_tokenizer.batch_decode(prompts_tokens.input_ids[0])))
                self.has_pri_decode = True
                
            replaced_idx = torch.nonzero(prompts_tokens.input_ids==unk_token_id)
            if not self.use_lora:
                prompt_embeds = self.llama_model.model.embed_tokens(prompts_tokens.input_ids)
            else:
                prompt_embeds = self.llama_model.base_model.model.model.embed_tokens(prompts_tokens.input_ids)
            if "<uid>" in prompt_ori  and "<his>" in prompt_ori and  "<iid>" in prompt_ori:
                prompt_embeds[replaced_idx[:,0],replaced_idx[:,1]] = samples['merged_embs']
            elif "<uid>" in prompt_ori and "<iid>" in prompt_ori and "<his>" not in prompt_ori:
                new_embeds = torch.cat([samples['User_emb'], samples['TargetItem_emb']], dim=-2).reshape(-1, samples['User_emb'].shape[-1])
                new_embeds = new_embeds.to(prompt_embeds.dtype)
                prompt_embeds = prompt_embeds.clone()
                prompt_embeds[replaced_idx[:, 0], replaced_idx[:, 1]] = new_embeds
            elif "<DCNFeature>" in prompt_ori:
                prompt_embeds[replaced_idx[:,0],replaced_idx[:,1]] = samples['User_emb'].reshape(-1,samples['User_emb'].shape[-1])
            else:
                pass 
            return prompt_embeds, prompts_tokens.attention_mask

    def forward(self,samples, only_logit=False):
        if self.run_mode_ == 'v2':
            return self.forward_v2(samples, only_logit=only_logit)
        if self.run_mode_ == 'v3':
            return self.forward_v3(samples, only_logit=only_logit)
        else:
            raise NotImplementedError("None-template version has not been implemtned...")  

    def prompt_based_encode_v2(self,prompt, samples):
        id_orders = get_ids_order(prompt) 
        samples_encode, atts_samples = self.encode_recdata_v2(samples,ids_order=id_orders)
        sample_embeds, atts_samples = self.recprompt_wrap_v2(samples_encode, samples, atts_samples, prompt) # 这里处理把title等装进去
        return sample_embeds, atts_samples
    
    def prompt_with_p(self,p):
        if self.prompt_list_p is None:
            prompt_list_p= []
            for k in range(len(p)):
                prompt_list_p.extend([self.prompt_list[k]]*p[k])
            self.prompt_list_p = prompt_list_p
            return self.prompt_list_p
        else:
            return self.prompt_list_p
    def maybe_autocast(self, dtype=torch.bfloat16):
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()
        
    def forward_v2(self, samples, only_logit=False):
        if hasattr(samples, 'question_split'):  
            print('VQA Batch')
            raise NotImplementedError("not implement")
        elif self.prompt_list:
            prompt = random.choice(self.prompt_list)
            sample_embeds, atts_samples = self.prompt_based_encode_v2(prompt,samples)
        self.llama_tokenizer.padding_side = "right"
        device = samples['uid'].device #samples_encode['User_emb'].device
        ans_ = {1:self.pos_ans[0], 0:self.neg_ans[0]}
        text = [ans_[int(t)] for t in samples["label"]] 
        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(device)

        t_posi = to_regress_tokens.input_ids.shape[-1] + 1

        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )
        empty_targets = torch.ones([atts_samples.shape[0],atts_samples.shape[1]],dtype=torch.long).to(device).fill_(-100)
        targets = torch.cat([empty_targets, targets], dim=1)
        if not self.use_lora:
            to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids)
        else:
            to_regress_embeds = self.llama_model.base_model.model.model.embed_tokens(to_regress_tokens.input_ids)
        inputs_embeds = torch.cat([sample_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_samples, to_regress_tokens.attention_mask], dim=1)

        with self.maybe_autocast():
            outputs = self.llama_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                    labels=targets,
                )

        pos_ans_id = self.llama_tokenizer(ans_[int(1)],add_special_tokens=False).input_ids[0]
        neg_ans_id = self.llama_tokenizer(ans_[int(0)],add_special_tokens=False).input_ids[0]
        logits = outputs.logits[:,-t_posi,:][:,pos_ans_id]
        if only_logit:
            return {'logits':logits}
        else:
            loss = nn.functional.binary_cross_entropy_with_logits(logits, samples['label'].float()) 
            return {"loss": loss}

    def forward_v3(self, samples, only_logit=False):
        if hasattr(samples, 'question_split'):  
            print('VQA Batch')
            raise NotImplementedError("not implement")
        elif self.prompt_list:
            prompt = random.choice(self.prompt_list) # (self.prompt_with_p([5,5,5,1])) #[1,5,3,1]  #[2,5,3,1]
            sample_embeds, atts_samples = self.prompt_based_encode_v2(prompt,samples)
        self.llama_tokenizer.padding_side = "right"
        device = samples['uid'].device #samples_encode['User_emb'].device
        ans_ = {1:self.pos_ans[0], 0:self.neg_ans[0]}
        text = [ans_[int(t)] for t in samples["label"]] 
        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(device)

        t_posi = to_regress_tokens.input_ids.shape[-1] + 1

        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )
        empty_targets = torch.ones([atts_samples.shape[0],atts_samples.shape[1]],dtype=torch.long).to(device).fill_(-100)
        targets = torch.cat([empty_targets, targets], dim=1)
        if not self.use_lora:
            to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids)
        else:
            to_regress_embeds = self.llama_model.base_model.model.model.embed_tokens(to_regress_tokens.input_ids)
        inputs_embeds = torch.cat([sample_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_samples, to_regress_tokens.attention_mask], dim=1)
        # if not self.config.use_gradient_checkpoint:
        with self.maybe_autocast():
            outputs = self.llama_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                    labels=targets,
                )

        # new loss, just focus on the target pos and neg tokens 
        if not self.config.use_logit_softmax:
            pos_ans_id = self.llama_tokenizer(ans_[int(1)],add_special_tokens=False).input_ids[0]
            neg_ans_id = self.llama_tokenizer(ans_[int(0)],add_special_tokens=False).input_ids[0]
            pos_logits = outputs.logits[:,-t_posi,:][:,pos_ans_id]
            neg_logits = outputs.logits[:,-t_posi,:][:,neg_ans_id]
        else:
            output_logits = outputs.logits.softmax(dim = -1)
            pos_ans_id = self.llama_tokenizer(ans_[int(1)],add_special_tokens=False).input_ids[0]
            neg_ans_id = self.llama_tokenizer(ans_[int(0)],add_special_tokens=False).input_ids[0]
            pos_logits = output_logits[:,-t_posi,:][:,pos_ans_id]
            neg_logits = output_logits[:,-t_posi,:][:,neg_ans_id]

        logits = torch.div(torch.exp(pos_logits) , (torch.exp(pos_logits) + torch.exp(neg_logits)))
        if only_logit:
            return {'logits':logits}
        else:
            loss = nn.functional.binary_cross_entropy(logits, samples['label'].to(torch.bfloat16)) 
            return {"loss": loss}

    def generate_for_samples_v2(self, samples,return_all=False):
        user_selective_prompts = False
        if hasattr(samples, 'question_split'): 
            print('VQA Batch')
            raise NotImplementedError("not implement")
        prompt = self.prompt_list[0]
        sample_embeds, atts_samples = self.prompt_based_encode_v2(prompt,samples)
        self.llama_tokenizer.padding_side = "right"
        device = samples['uid'].device #samples_encode['User_emb'].device
        pos_ans = self.pos_ans[0]
        neg_ans = self.neg_ans[0]
        ans_ = {1:pos_ans, 0:neg_ans}
        text = [ ans_[int(t)]  for t in samples["label"]]
        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(device)
        t_posi = to_regress_tokens.input_ids.shape[-1] + 1
        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )
        empty_targets = torch.ones([atts_samples.shape[0],atts_samples.shape[1]],dtype=torch.long).to(device).fill_(-100)
        targets = torch.cat([empty_targets, targets], dim=1)

        if not self.use_lora:
            to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids)
        else:
            to_regress_embeds = self.llama_model.base_model.model.model.embed_tokens(to_regress_tokens.input_ids)
        inputs_embeds = torch.cat([sample_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_samples, to_regress_tokens.attention_mask], dim=1)

        with self.maybe_autocast():
            outputs = self.llama_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                    labels=targets,
                )
        pos_ans_id = self.llama_tokenizer(pos_ans, add_special_tokens=False).input_ids[0]
        neg_ans_id = self.llama_tokenizer(neg_ans, add_special_tokens=False).input_ids[0]
        logits_ = outputs.logits[:,-t_posi,:][:,pos_ans_id]
        loss = nn.functional.binary_cross_entropy_with_logits(logits_, samples['label'].float())
        if return_all:
            return outputs, logits_
        return {"loss": loss, 'logits':logits_}

    
    def generate_for_samples_v3(self, samples,return_all=False):
        user_selective_prompts = False
        if hasattr(samples, 'question_split'): 
            print('VQA Batch')
            raise NotImplementedError("not implement")

        prompt = self.prompt_list[0]
        sample_embeds, atts_samples = self.prompt_based_encode_v2(prompt,samples)
        self.llama_tokenizer.padding_side = "right"
        device = samples['uid'].device #samples_encode['User_emb'].device
        pos_ans = self.pos_ans[0]
        neg_ans = self.neg_ans[0]
        ans_ = {1:pos_ans, 0:neg_ans}
        text = [ ans_[int(t)]  for t in samples["label"]]
        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(device)
        t_posi = to_regress_tokens.input_ids.shape[-1] + 1
        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )
        empty_targets = torch.ones([atts_samples.shape[0],atts_samples.shape[1]],dtype=torch.long).to(device).fill_(-100)
        targets = torch.cat([empty_targets, targets], dim=1)

        if not self.use_lora:
            to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids)
        else:
            to_regress_embeds = self.llama_model.base_model.model.model.embed_tokens(to_regress_tokens.input_ids)
        inputs_embeds = torch.cat([sample_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_samples, to_regress_tokens.attention_mask], dim=1)

        with self.maybe_autocast():
            outputs = self.llama_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                    labels=targets,
                )
        if not self.config.use_logit_softmax:
            pos_ans_id = self.llama_tokenizer(ans_[int(1)],add_special_tokens=False).input_ids[0]
            neg_ans_id = self.llama_tokenizer(ans_[int(0)],add_special_tokens=False).input_ids[0]
            pos_logits = outputs.logits[:,-t_posi,:][:,pos_ans_id]
            neg_logits = outputs.logits[:,-t_posi,:][:,neg_ans_id]
        else:
            output_logits = outputs.logits.softmax(dim = -1)
            pos_ans_id = self.llama_tokenizer(ans_[int(1)],add_special_tokens=False).input_ids[0]
            neg_ans_id = self.llama_tokenizer(ans_[int(0)],add_special_tokens=False).input_ids[0]
            pos_logits = output_logits[:,-t_posi,:][:,pos_ans_id]
            neg_logits = output_logits[:,-t_posi,:][:,neg_ans_id]
        logits = torch.div(torch.exp(pos_logits) , (torch.exp(pos_logits) + torch.exp(neg_logits)))
        loss = nn.functional.binary_cross_entropy(logits, samples['label'].to(torch.bfloat16)) 

        if return_all:
            return outputs, logits
        return {"loss": loss, 'logits':logits}

    def generate_for_samples(self,samples, return_all = False):
        if self.run_mode_ == 'v2':
            return self.generate_for_samples_v2(samples, return_all)
        if self.run_mode_ == 'v3':
            return self.generate_for_samples_v3(samples, return_all)
        else:
            raise NotImplementedError("Not implement the default version")     