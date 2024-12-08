import datetime
import json
import logging
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd

import torch
import torch.distributed as dist
import webdataset as wds
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from utils import *
from omegaconf import OmegaConf
from datasets import IFdatasets
import copy
from tqdm import tqdm
from calculate_if import *
from peft import *
import bitsandbytes as bnb
class Runner:
    def __init__(self, cfg, model, datasets, ref_model = None):
        self.config = cfg
        self.datasets = datasets # dataset builder
        self._model = model
        self._wrapped_model = None
        self._device = None
        self._optimizer = None
        self._scaler = None
        self._dataloaders = None
        self._lr_sched = None
        self._ref_model = ref_model
        self.start_epoch = 0
        self.setup_output_dir()
        self.setup_save_path()

    def setup_output_dir(self):

        output_dir = self.config.output_dir 
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir

    def setup_save_path(self):
        self.generate_synthetic = self.config.generate_synthetic
        self.save_synthetic_path = self.config.get('save_synthetic_path', None)
        self.save_test_path = self.config.get('save_test_path', None)

        if self.save_synthetic_path:
            os.makedirs(os.path.dirname(self.save_synthetic_path), exist_ok=True)
        if self.save_test_path:
            os.makedirs(os.path.dirname(self.save_test_path), exist_ok=True)


    @property
    def resume_ckpt_path(self):
        return self.config.get("resume_ckpt_path", None)
    
    def set_model_mode(self,mode):
        if self.use_distributed:
            self.model.module.set_mode(mode)
        else:
            self.model.set_mode(mode)

    @main_process
    def log_stats(self, stats, split_name):
        def convert_to_builtin_type(obj):
            if isinstance(obj, np.int64):
                return int(obj)
        if isinstance(stats, dict):
            log_stats = {**{f"{split_name}_{k}": v for k, v in stats.items()}}
            with open(os.path.join(self.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats, default = convert_to_builtin_type) + "\n")
        elif isinstance(stats, list):
            pass

    @main_process
    def log_config(self):
        with open(os.path.join(self.output_dir, "log.txt"), "a") as f:
            f.write(json.dumps(OmegaConf.to_container(self.config), indent=4) + "\n")
    @property
    def evaluate_only(self):
        """
        Set to True to skip training.
        """
        return self.config.evaluate_only
    
    @property
    def max_epoch(self):
        return int(self.config.max_epoch)
    
    @property
    def use_distributed(self):
        return self.config.distributed    
    
    def model_to_betrained(self):
        if self.use_distributed:
            return self.model.module.to_be_trained()
        else:
            return self.model.to_be_trained()    
        
    @property
    def model(self):
        """
        A property to get the DDP-wrapped model on the device.
        """
        # move model to device
        if self._model.device != self.device:
            self._model = self._model.to(self.device)

            # distributed training wrapper
            if self.use_distributed:
                if self._wrapped_model is None:
                    self._wrapped_model = DDP(
                        self._model, device_ids=[self.config.gpu], find_unused_parameters=True
                    )
            else:
                self._wrapped_model = self._model

        return self._wrapped_model

    @property
    def device(self):
        if self._device is None:
            self._device = torch.device(self.config.device)

        return self._device

    def get_rec_model_loss(self,dataset, batch_size):
        data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=self.config.num_workers, shuffle=False)
        total_loss = 0
        for samples in data_loader:
            samples = prepare_sample(samples)
            rec_logits = self.model.if_rec_forward(samples).float().squeeze()
            labels = samples['label'].float().squeeze()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(rec_logits, labels, reduction='sum')
            total_loss += loss
        total_loss /= len(data_loader)
        return total_loss

    def fit(self):
        return self.ref_fit()

    def ref_fit(self):
        start_time = time.time()
        best_agg_metric = -100000
        best_epoch = 0
        not_change = 0
        self.set_model_mode(self.config.mode)
        self.log_config()

        if not self.evaluate_only:# with training
            for cur_epoch in range(self.start_epoch, self.max_epoch):
                if not self.evaluate_only and self.model_to_betrained():
                    logging.info("Start training")
                    # having lora or IDs are used
                    if self.config.use_gradient_checkpoint:
                        self.model.llama_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={})
                    train_stats = self.train_epoch(cur_epoch)
                    self.log_stats(split_name="train", stats=train_stats)
                
                if len(self.valid_splits) > 0:
                    for split_name in self.valid_splits:
                        logging.info("Evaluating on {}.".format(split_name))
                        val_log = self.eval_epoch(
                            split_name=split_name, cur_epoch=cur_epoch
                        )
                        
                        if val_log is not None:
                            if is_main_process():
                                assert (
                                    "agg_metrics" in val_log
                                ), "No agg_metrics found in validation log."
                                agg_metrics = val_log["agg_metrics"]
                                if agg_metrics > best_agg_metric and split_name == "valid":
                                    best_epoch, best_agg_metric = cur_epoch, agg_metrics
                                    self._save_checkpoint(cur_epoch, is_best=True)
                                    not_change = 0
                                val_log.update({"best_epoch": best_epoch})
                                self.log_stats(val_log, split_name)
                                not_change += 1
                                # if not_change > 20: # early stop
                                #     break
                        # torch.cuda.empty_cache()
                else:
                    if not self.evaluate_only:
                        self._save_checkpoint(cur_epoch, is_best=False)
                if self.evaluate_only:
                    break
                if self.config.distributed:
                    dist.barrier()
                if not self.model_to_betrained():
                    break
                if not_change > self.config.patience:
                    logging.info("Early stop. The results has not changed up to 10 epochs.")
                    break

        # testing phase, would only run when evaluate_only==True
        if self.evaluate_only:
            print("training finish or just evaluation...")
            if len(self.test_splits) > 0:
                logging.info("Evaluating on {}.".format(self.test_splits[0]))
            test_epoch = "best" if len(self.valid_splits) > 0 else cur_epoch
            # generate synthetic datasets:
            if self.generate_synthetic:
                self.generate_synthetic_data()
            else:
                print('do not generate synthetic interactions because the set generate_synthetic False')

            # save evaluation metrics 
            if len(self.test_splits) > 0:
                result = self.evaluate(cur_epoch=test_epoch, skip_reload=self.evaluate_only)
                for key, value in result.items():
                    try:
                        result[key]['uauc'] = result[key]['uauc'][0]
                    except:
                        pass
                def default(o):
                    if isinstance(o, np.int64):
                        return int(o)
                    return None
                if self.save_test_path is not None:
                    with open(self.save_test_path, 'w') as json_f:
                        json.dump(result, json_f, default = default, indent=2)
                else:
                    print('do not save evaluation results because save test path is None')
                        # json_f.write()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logging.info("Training time {}".format(total_time_str))
        self.set_model_mode(None) # recover to the default model




    # generate synthetic data and save it
    def generate_synthetic_data(self):
        if self.synthetic_splits is not None:
            if self.synthetic_splits is not []:
                data_loader = self.synthetic_loader
                assert data_loader, "synthetic split is not None, but synthetic data_loader is None."
            else:
                print('do not generate synthetic data because the synthetic splits is None')
                return
        else:
            print('do not generate synthetic data because the synthetic splits is None')
            return
        

        model = self.unwrap_dist_model(self.model)
        model.eval()

        evaluated_dataloader = data_loader
        save_data_list = []
        for i, samples in enumerate(evaluated_dataloader ):
            samples =  prepare_sample(samples)
            with torch.no_grad():
                eval_output, eval_logits = model.generate_for_samples(samples=samples, return_all = True)
            # print(samples)
            # print(eval_logits)
            batch_data_dict = {}
            batch_data_dict['uid'] = samples['uid'].clone().detach().cpu().numpy().tolist()
            batch_data_dict['iid'] = samples['iid'].clone().detach().cpu().numpy().tolist()
            batch_data_dict['his'] = [his_id.tolist() for his_id in samples['his_pad'].clone().detach().cpu().numpy()] # todo 确认数据加载。
            batch_data_dict['predicted_logit'] = eval_logits.clone().detach().cpu().float().numpy().tolist()
            batch_df = pd.DataFrame(batch_data_dict)
            save_data_list.append(batch_df)
        # print(save_data_list)
        generated_data = pd.concat(save_data_list)
        generated_data.to_pickle(self.save_synthetic_path)
        return
    
    @property
    def optimizer(self):
        # TODO make optimizer class and configurations
        if self._optimizer is None:
            num_parameters = 0
            p_wd, p_non_wd = [], []
            for n, p in self.model.named_parameters():
                if not p.requires_grad:
                    continue  # frozen weights
                if p.ndim < 2 or "bias" in n or "ln" in n or "bn" in n:
                    p_non_wd.append(p)
                else:
                    p_wd.append(p)
                num_parameters += p.data.nelement()
            logging.info("number of trainable parameters: %d" % num_parameters)
            self._num_trainable_para = num_parameters > 0
            optim_params = [
                {
                    "params": p_wd,
                    "weight_decay": float(self.config.weight_decay),
                },
                {"params": p_non_wd, "weight_decay": 0},
            ]
            beta2 = self.config.get("beta2", 0.999)
            self._optimizer = torch.optim.AdamW(
                optim_params,
                lr=float(self.config.init_lr),
                weight_decay=float(self.config.weight_decay),
                betas=(0.9, beta2),
            )
            if self.config.use_8bit_optim:
            # if self.config.ref_full and not self.config.run_dpo:
                self._optimizer = bnb.optim.AdamW(
                    optim_params,
                    lr=float(self.config.init_lr),
                    weight_decay=float(self.config.weight_decay),
                    betas=(0.9, beta2),
                    optim_bits=8,
                )

        return self._optimizer
    
    
    @property
    def scaler(self):
        amp = self.config.get("amp", False)

        if amp:
            if self._scaler is None:
                self._scaler = torch.cuda.amp.GradScaler()

        return self._scaler

    @property
    def lr_scheduler(self):
        """
        A property to get and create learning rate scheduler by split just in need.
        """
        if self._lr_sched is None:
            lr_sched_cls = LinearWarmupCosineLRScheduler
            # max_epoch = self.config.max_epoch
            max_epoch = self.max_epoch
            # min_lr = self.config.min_lr
            min_lr = self.min_lr
            # init_lr = self.config.init_lr
            init_lr = self.init_lr

            # optional parameters
            decay_rate = self.config.get("lr_decay_rate", None)
            warmup_start_lr = self.config.get("warmup_lr", -1)
            warmup_steps = self.config.get("warmup_steps", 0)
            iters_per_epoch = self.config.get("iters_per_epoch", None)

            if iters_per_epoch is None:
                try:
                    iters_per_epoch = len(self.dataloaders['train'])
                except (AttributeError, TypeError):
                    iters_per_epoch = 10000

            self._lr_sched = lr_sched_cls(
                optimizer=self.optimizer,
                max_epoch=max_epoch,
                iters_per_epoch=iters_per_epoch,
                min_lr=min_lr,
                init_lr=init_lr,
                decay_rate=decay_rate,
                warmup_start_lr=warmup_start_lr,
                warmup_steps=warmup_steps,
            )

        return self._lr_sched

    @property
    def dataloaders(self) -> dict:
        """
        A property to get and create dataloaders by split just in need.

        If no train_dataset_ratio is provided, concatenate map-style datasets and
        chain wds.DataPipe datasets separately. Training set becomes a tuple
        (ConcatDataset, ChainDataset), both are optional but at least one of them is
        required. The resultant ConcatDataset and ChainDataset will be sampled evenly.

        If train_dataset_ratio is provided, create a MultiIterLoader to sample
        each dataset by ratios during training.

        Currently do not support multiple datasets for validation and test.

        Returns:
            dict: {split_name: (tuples of) dataloader}
        """
        if self._dataloaders is None:

            _dataloaders = {}
            for split_name, dataset in self.datasets.items():
                is_train = split_name in self.train_splits
                batch_size = self.config.batch_size_train if split_name == "train" else self.config.batch_size_eval

                _dataloaders[split_name] = DataLoader(dataset, batch_size=batch_size,
                 num_workers=self.config.num_workers, shuffle=is_train)
            self._dataloaders = _dataloaders


        return self._dataloaders    
    
    def create_loaders(
        self,
        datasets,
        num_workers,
        batch_sizes,
        is_trains,
        collate_fns,
        dataset_ratios=None,
    ):
        """
        Create dataloaders for training and validation.
        """

        def _create_loader(dataset, num_workers, bsz, is_train, collate_fn):
            # create a single dataloader for each split
            if isinstance(dataset, ChainDataset) or isinstance(
                dataset, wds.DataPipeline
            ):
                # wds.WebdDataset instance are chained together
                # webdataset.DataPipeline has its own sampler and collate_fn
                loader = iter(
                    DataLoader(
                        dataset,
                        batch_size=bsz,
                        num_workers=num_workers,
                        pin_memory=True,
                    )
                )
            else:
                # map-style dataset are concatenated together
                # setup distributed sampler
                if self.use_distributed:
                    sampler = DistributedSampler(
                        dataset,
                        shuffle=is_train,
                        num_replicas=get_world_size(),
                        rank=get_rank(),
                    )
                    if not self.use_dist_eval_sampler:
                        # e.g. retrieval evaluation
                        sampler = sampler if is_train else None
                else:
                    sampler = None

                loader = DataLoader(
                    dataset,
                    batch_size=bsz,
                    num_workers=num_workers,
                    pin_memory=True,
                    sampler=sampler,
                    shuffle=sampler is None and is_train,
                    collate_fn=collate_fn,
                    drop_last=True if is_train else False,
                )
                loader = PrefetchLoader(loader)

                if is_train:
                    loader = IterLoader(loader, use_distributed=self.use_distributed)

            return loader

        loaders = []

        for dataset, bsz, is_train, collate_fn in zip(
            datasets, batch_sizes, is_trains, collate_fns
        ):
            if isinstance(dataset, list) or isinstance(dataset, tuple):
                if hasattr(dataset[0], 'sample_ratio') and dataset_ratios is None:
                    dataset_ratios = [d.sample_ratio for d in dataset]
                loader = MultiIterLoader(
                    loaders=[
                        _create_loader(d, num_workers, bsz, is_train, collate_fn[i])
                        for i, d in enumerate(dataset)
                    ],
                    ratios=dataset_ratios,
                )
            else:
                loader = _create_loader(dataset, num_workers, bsz, is_train, collate_fn)

            loaders.append(loader)

        return loaders

    @property
    def cuda_enabled(self):
        return self.device.type == "cuda"    
    @property
    def log_freq(self):
        log_freq = self.config.get("log_freq", 50)
        return int(log_freq)

    @property
    def init_lr(self):
        return float(self.config.init_lr)

    @property
    def min_lr(self):
        return float(self.config.min_lr)

    @property
    def accum_grad_iters(self):
        return int(self.config.get("accum_grad_iters", 1))

    @property
    def train_loader(self):
        train_dataloader = self.dataloaders.get('train', None)
        return train_dataloader
    
    @property
    def valid_splits(self):
        valid_splits = self.config.get("valid_splits", [])

        if len(valid_splits) == 0:
            logging.info("No validation splits found.")

        return valid_splits        
    
    @property
    def test_splits(self):
        test_splits = self.config.get("test_splits", [])

        return test_splits

    @property  
    def train_splits(self):
        train_splits = self.config.get("train_splits", [])

        if len(train_splits) == 0:
            logging.info("Empty train splits.")

        return train_splits
    
    @property  
    def synthetic_splits(self):
        synthetic_splits = self.config.get("synthetic_splits", [])
        return synthetic_splits
    
    @property
    def synthetic_loader(self):
        synthetic_dataloader = self.dataloaders.get('synthetic', None)
        return synthetic_dataloader

    def train_step(self, model, samples):
        loss = model(samples)["loss"]
        return loss
    
    
    def train_epoch(
        self, epoch, scaler=None, start_iters=None, log_freq=50, cuda_enabled=False, accum_grad_iters=1):
        self.model.train()
        model=self.model
        data_loader=self.train_loader  # todo 我不想用train_split的参数，而是用单独property
        optimizer=self.optimizer
        scaler=self.scaler
        lr_scheduler=self.lr_scheduler
        cuda_enabled=self.cuda_enabled
        log_freq=self.log_freq
        accum_grad_iters=self.accum_grad_iters
        iters_per_epoch = self.lr_scheduler.iters_per_epoch
        use_amp = False
        # scaler is not None # False
        if not hasattr(data_loader, "__next__"):
            # convert to iterator if not already
            data_loader = iter(data_loader)

        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))

        # if iter-based runner, schedule lr based on inner epoch.
        logging.info(
            "Start training epoch {}, {} iters per inner epoch.".format(
                epoch, iters_per_epoch
            )
        )
        header = "Train: data epoch: [{}]".format(epoch)
        if start_iters is None:
            # epoch-based runner
            inner_epoch = epoch
        else:
            # In iter-based runner, we schedule the learning rate based on iterations.
            inner_epoch = start_iters // iters_per_epoch
            header = header + "; inner epoch [{}]".format(inner_epoch)

        for i in metric_logger.log_every(range(iters_per_epoch), log_freq, header):
            # if using iter-based runner, we stop after iters_per_epoch iterations.
            if i >= iters_per_epoch:
                break
            samples = next(data_loader)
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
            lr_scheduler.step(cur_epoch=inner_epoch, cur_step=i)

            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = self.train_step(model=model, samples=samples)
            # after_train_step()
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # update gradients every accum_grad_iters iterations
            if (i + 1) % accum_grad_iters == 0:
                if use_amp:
                    # with torch.cuda.amp.autocast(enabled=False):
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()                     
                else:    
                    optimizer.step()
                optimizer.zero_grad()

            metric_logger.update(loss=loss.item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            torch.cuda.empty_cache()

        # after train_epoch()
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        logging.info("Averaged stats: " + str(metric_logger.global_avg()))
        return {
            k: "{:.3f}".format(meter.global_avg)
            for k, meter in metric_logger.meters.items()
        }


    @torch.no_grad()
    def eval_epoch(self, split_name, cur_epoch, skip_reload=False):
        """
        Evaluate the model on a given split.

        Args:
            split_name (str): name of the split to evaluate on.
            cur_epoch (int): current epoch.
            skip_reload_best (bool): whether to skip reloading the best checkpoint.
                During training, we will reload the best checkpoint for validation.
                During testing, we will use provided weights and skip reloading the best checkpoint .
        """
        data_loader = self.dataloaders.get(split_name, None)  
        assert data_loader, "data_loader for split {} is None.".format(split_name)

        model = self.unwrap_dist_model(self.model)
        if not skip_reload and cur_epoch == "best":
            model = self._reload_best_model(model)
        model.eval()

        results = self.evaluation(model, data_loader)

        return results # , cur_epoch)
    
    def unwrap_dist_model(self, model):
        if self.use_distributed:
            return model.module
        else:
            return model
        
    def _reload_best_model(self, model):
        """
        Load the best checkpoint for evaluation.
        """
        checkpoint_path = os.path.join(self.output_dir, "checkpoint_best.pth")

        logging.info("Loading checkpoint from {}.".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        try:
            model.load_state_dict(checkpoint["model"])
        except RuntimeError as e:
            logging.warning(
                """
                Key mismatch when loading checkpoint. This is expected if only part of the model is saved.
                Trying to load the model with strict=False.
                """
            )
            model.load_state_dict(checkpoint["model"], strict=False)
        return model
    
    @main_process
    def _save_checkpoint(self, cur_epoch, is_best=False):
        """
        Save the checkpoint at the current epoch.
        """
        model_no_ddp = self.unwrap_dist_model(self.model)
        param_grad_dic = {
            k: v.requires_grad for (k, v) in model_no_ddp.named_parameters()
        }
        state_dict = model_no_ddp.state_dict()
        proj_dict = {key: value for key, value in state_dict.items() if 'llama_proj' in key and 'lora' not in key}
        lora_dict = {key: value for key, value in state_dict.items() if 'lora' in key}


        lora = get_peft_model_state_dict(model_no_ddp.llama_model)
        # {key: value for key, value in state_dict.items() if 'lora' in key}
        p_count = len(state_dict)+0.0
        for k in list(state_dict.keys()):
            p_count += 1
            if k in param_grad_dic.keys() and not param_grad_dic[k]:
                # delete parameters that do not require gradient
                del state_dict[k]
        logging.info("when saving, the saving ratio is: {:.5f}".format(len(state_dict)/p_count))


        save_obj = {
            "lora": lora, # lora_dict,
            'lora_dict':lora_dict,
            "proj": proj_dict,
            "optimizer": self.optimizer.state_dict(),
            "config": OmegaConf.to_container(self.config),
            "scaler": self.scaler.state_dict() if self.scaler else None,
            "epoch": cur_epoch,
        }
        save_to = os.path.join(
            self.output_dir,
            "checkpoint_{}.pth".format("best" if is_best else cur_epoch),
        )
        logging.info("Saving checkpoint at epoch {} to {}.".format(cur_epoch, save_to))
        torch.save(save_obj, save_to)            
    # evaluate step
    def evaluation(self, model, data_loader, cuda_enabled=True):
        model = model.eval()
        metric_logger = MetricLogger(delimiter="  ")
        auc_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))
        metric_logger.add_meter("acc", SmoothedValue(window_size=1, fmt="{value:.4f}"))
        auc_logger.add_meter("auc", SmoothedValue(window_size=1, fmt="{value:.4f}"))
        header = "Evaluation"
        print_freq = 10
        results = []
        results_logits = []
        labels = []
        users = []
        # j = 0
        for samples in metric_logger.log_every(data_loader, print_freq, header):
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
            eval_output = self.valid_step(model=model, samples=samples)
            # results_loss.append(eval_output['loss'].item())
            if 'logits' in eval_output.keys():
                use_auc = True
                users.extend(samples['uid'].detach().cpu().numpy())
                results_logits.extend(eval_output['logits'].detach().cpu().float().numpy())
                labels.extend(samples['label'].detach().cpu().numpy())
                logits = eval_output['logits']
                logits[logits>0.5] = 1
                acc = (logits-samples['label'])
                acc = (acc==0).sum()/acc.shape[0]
                metric_logger.update(acc=acc.item())
            else: 
                metric_logger.update(acc=0)
            metric_logger.update(loss=eval_output['loss'].item())
            
        results_logits_ = torch.tensor(results_logits).to(eval_output['logits'].device).contiguous()
        labels_ = torch.tensor(labels).to(eval_output['logits'].device).contiguous()
        users_ = torch.tensor(users).to(eval_output['logits'].device).contiguous()

        auc = 0
        if is_dist_avail_and_initialized():
            print("wating comput auc.....")
            rank = dist.get_rank()
            gathered_labels = [labels_.clone() for _ in range(dist.get_world_size())]
            gathered_logits = [results_logits_.clone() for _ in range(dist.get_world_size())]
            gathered_users = [users_.clone() for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_labels, labels_)
            dist.all_gather(gathered_logits, results_logits_)
            dist.all_gather(gathered_users, users_)
            
            labels_a = torch.cat(gathered_labels,dim=0).flatten().cpu().numpy()
            results_logits_a = torch.cat(gathered_logits,dim=0).flatten().cpu().numpy()
            users_a = torch.cat(gathered_users,dim=0).flatten().cpu().numpy()
            print("computing....")
            auc = roc_auc_score(labels_a, results_logits_a)
            uauc, _, _ = uAUC_me(users_a,results_logits_a,labels_a)
            print("finished comput auc.....")
        else:
            auc = roc_auc_score(labels_.cpu().numpy(), results_logits_.cpu().numpy())
            uauc = uAUC_me(users_.cpu().numpy(), results_logits_.cpu().numpy(), labels_.cpu().numpy())
        if use_auc:
            auc_rank0 = roc_auc_score(labels_.cpu().numpy(), results_logits_.cpu().numpy())
        logging.info("Averaged stats: " + str(metric_logger.global_avg()) + " ***auc: " + str(auc) + " ***uauc:" +str(uauc) )
        print("rank_0 auc:", str(auc_rank0))        
        if use_auc:
            results = {
                'agg_metrics':auc,
                'acc': metric_logger.meters['acc'].global_avg,
                'loss':  metric_logger.meters['loss'].global_avg,
                'uauc': uauc
            }
        else: # only loss usable
            results = {
                'agg_metrics': -metric_logger.meters['loss'].global_avg,
            }
        if is_dist_avail_and_initialized():
            dist.barrier()
        
        metric_logger.synchronize_between_processes()   

        return results

    # evaluate and get results
    def evaluate(self, cur_epoch="best", skip_reload=False):
        test_logs = dict()

        if len(self.test_splits) > 0:
            for split_name in self.test_splits:
                data_loader = self.dataloaders.get(split_name, None)
                assert data_loader, "data_loader for split {} is None.".format(split_name)
                test_logs[split_name] = self.eval_epoch(
                split_name=split_name, cur_epoch=cur_epoch, skip_reload=skip_reload
                )
            return test_logs
        
    def valid_step(self, model, samples):
        outputs = model.generate_for_samples(samples)
        return outputs


    def inference_step(self):
        raise NotImplementedError