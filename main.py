from __future__ import print_function

import matplotlib.pyplot as plt
import cv2
import os
from data.dataset import Dataset
import torch
from torch.utils import data
import torch.nn.functional as F
from models import *
import torch
import numpy as np
import time
import config.config as config

from models import *
from models.backbone.irsnet import iresnet50, iresnet100
import utils
from torch.utils.tensorboard import SummaryWriter
import torchinfo
import torch.utils.data

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import torch.multiprocessing 
import logging
from tqdm import tqdm

class FilteredDataset(data.Dataset):
    def __init__(self, original_dataset, classes_to_keep):
        self.original_dataset = original_dataset
        self.classes_to_keep = classes_to_keep # 사용할 클래스수
        self.class_mapping = {old_label: new_label for new_label, old_label in enumerate(self.classes_to_keep)} # {기존클래스 : 새로운클래스명}

        original_labels = np.array(self.original_dataset.labels)
        mask = np.isin(original_labels, self.classes_to_keep) # mask생성 [0,5,10,3] [5,10] -> [F,T,T,F]
        self.indices = np.where(mask)[0]

        self.remapped_labels = np.array([self.class_mapping[label] for label in original_labels[mask]])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        original_index = self.indices[index]
        image, _, path = self.original_dataset[original_index]
        new_label = self.remapped_labels[index]
        return image, new_label, path

    @property
    def get_classes(self):
        return len(self.classes_to_keep)

def save_model(backbone, metric_fc, save_path, name, iter_cnt):
    os.makedirs(save_path, exist_ok=True)
    save_name = os.path.join(save_path, name + '_' + str(iter_cnt) + '.pth')

    backbone_state_dict = backbone.module.state_dict() if hasattr(backbone, 'module') else backbone.state_dict()
    metric_fc_state_dict = metric_fc.module.state_dict() if hasattr(metric_fc, 'module') else metric_fc.state_dict()

    torch.save({
        'backbone_state_dict': backbone_state_dict,
        'metric_fc_state_dict': metric_fc_state_dict,
    }, save_name)

    logging.info(f"Model saved to {save_name}")
    return save_name

def load_weights(model, weight_path, rank, model_name):
    if weight_path and os.path.exists(weight_path):
        load_result = model.load_state_dict(
            torch.load(weight_path, map_location='cpu'),
            strict=False
        )
        if rank == 0:
            logging.info(f"Pre-trained weights for {model_name} loaded from {weight_path}.")
            if load_result.missing_keys or load_result.unexpected_keys:
                logging.info(f"Missing keys for {model_name}: {load_result.missing_keys}")
                logging.info(f"Unexpected keys for {model_name}: {load_result.unexpected_keys}")
    else:
        if rank == 0:
            logging.info(f"No pre-trained weights for {model_name} found or specified, starting from scratch.")

def cleanup_ddp():
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

def setup_ddp(rank, world_size, port=13355):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(port)
    
    torch.distributed.init_process_group(
        backend='nccl', 
        rank=rank, 
        world_size=world_size
    )
    torch.cuda.set_device(rank)
    
    print(f"Multi GPU SETUP - RANK: {rank}, WORLD_SIZE: {world_size}")
    return rank


def main():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True

    ngpus_per_node = torch.cuda.device_count()

    if ngpus_per_node < 2:
        logging.warning("Warning: Single GPU training detected.")

    torch.multiprocessing.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node,),join=True)

def main_worker(rank, world_size):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(f"GPU Available: {torch.cuda.device_count()}")
    device_id = setup_ddp(rank, world_size)

    logging.info(f"Process {rank} started on device {device_id}")


    logdir_name = None
    if rank == 0:
        log_idx = 0
        while True:
            logdir_name = f'./logs/train_{log_idx}'
            if not os.path.exists(logdir_name):
                os.makedirs(logdir_name, exist_ok=True)
                logging.info(f"Creating TensorBoard log directory: {logdir_name}")
                break
            log_idx += 1
    dist.barrier()

    writer = SummaryWriter(log_dir=logdir_name) if rank == 0 else None
    opt = config.get_config(rank)


    scaler = torch.amp.GradScaler(device=f'cuda:{device_id}')
    device = torch.device(f"cuda:{device_id}")

    original_full_dataset = Dataset(root=f'{opt.train_root}', phase='train', input_shape=(1, 112, 112))
    
    num_classes_to_keep = original_full_dataset.get_classes // 2
    
    if rank == 0:
        logging.info(f"Original number of classes: {original_full_dataset.get_classes}")
        logging.info(f"Keeping {num_classes_to_keep} classes for training.")
        classes_to_keep = np.random.choice(range(original_full_dataset.get_classes), num_classes_to_keep, replace=False)
        classes_to_keep_tensor = torch.from_numpy(classes_to_keep).to(device)
    else:
        classes_to_keep_tensor = torch.empty(num_classes_to_keep, dtype=torch.long, device=device)

    dist.broadcast(classes_to_keep_tensor, src=0)

    classes_to_keep = classes_to_keep_tensor.cpu().numpy()
    filtered_dataset = FilteredDataset(original_full_dataset, classes_to_keep)

    train_size = int(0.8 * len(filtered_dataset))
    val_size = len(filtered_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(filtered_dataset, [train_size, val_size])

    train_sampler = torch.utils.data.DistributedSampler(
        train_dataset, 
        num_replicas=world_size, 
        rank=rank, 
        shuffle=True
    )
    val_sampler = torch.utils.data.DistributedSampler(
        val_dataset, 
        num_replicas=world_size, 
        rank=rank, 
        shuffle=False
    )

    trainloader = data.DataLoader(
        train_dataset,
        batch_size=opt.train_batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        sampler=train_sampler,
        pin_memory=True
    )

    valloader = data.DataLoader(
        val_dataset,
        batch_size=opt.train_batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        sampler=val_sampler,
        pin_memory=True
    )

    if rank == 0:
        logging.info(f"Training set size: {len(train_dataset)}")
        logging.info(f"Validation set size: {len(val_dataset)}")


    if opt.loss == 'focal_loss': # 손실함수 선택
        criterion = FocalLoss(gamma=2) # 클래스불균형시
    else:
        criterion = torch.nn.CrossEntropyLoss()

    """    Backbone  선택 """
    if opt.backbone == 'iresnet50':
        backbone = iresnet50()
        if rank == 0:
            logging.info("Using IResNet-50 backbone")
    elif opt.backbone == 'iresnet100':
        backbone = iresnet100()
        if rank == 0:
            logging.info("Using IResNet-100 backbone")
    else:
        backbone = iresnet50()
        if rank == 0:
            logging.info("Defaulting to IResNet-50 backbone")

    

    for param in backbone.parameters():
        param.requires_grad = False

    num_classes = filtered_dataset.get_classes 

    """ HEAD """
    if opt.metric == 'add_margin':
        metric_fc = AddMarginProduct(512, num_classes, s=30, m=0.35)
    elif opt.metric == 'arc_margin':
        metric_fc = ArcMarginProduct(512, num_classes, s=30, m=0.5, easy_margin=opt.easy_margin)
    elif opt.metric == 'sphere':
        metric_fc = SphereProduct(512, num_classes, m=4)
    else:
        metric_fc = ArcMarginProduct(512, num_classes, s=30, m=0.5, easy_margin=opt.easy_margin)
 
    
    load_weights(backbone, opt.backbone_pretrained_weights, rank, "Backbone")
    load_weights(metric_fc, opt.head_pretrained_weight, rank, "Head")


    if rank == 0:
        backbone_params = sum(p.numel() for p in backbone.parameters()) #  백본 파라매터
        trainable_backbone_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
        logging.info('==' * 30)
        logging.info(f"Backbone total params: {backbone_params:,}")
        logging.info(f"Backbone trainable params: {trainable_backbone_params:,}")
        logging.info(f"Head total params: {sum(p.numel() for p in metric_fc.parameters()):,} ")
        logging.info(f"Head trainable params: {sum(p.numel() for p in metric_fc.parameters() if p.requires_grad):,}")
        logging.info('==' * 30)

    backbone.to(device)
    metric_fc.to(device)

    if sum(p.numel() for p in backbone.parameters() if p.requires_grad) > 0: # 백본 동결시 분산x
        backbone = DistributedDataParallel(backbone, device_ids=[device_id], find_unused_parameters=False)
    metric_fc = DistributedDataParallel(metric_fc, device_ids=[device_id], find_unused_parameters=False)

    #backbone = torch.compile(backbone, options={"max_autotune": False})



    if opt.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            metric_fc.parameters(), 
            lr=opt.lr, 
            weight_decay=opt.weight_decay
        )
    elif opt.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            metric_fc.parameters(),
            lr=opt.lr,
            weight_decay=opt.weight_decay
        )
    else:  # adamw
        optimizer = torch.optim.AdamW(
            metric_fc.parameters(),
            lr=opt.lr,
            weight_decay=opt.weight_decay
        )


    scheduler = utils.lr_scheduler.PolynomialLRWarmup(
        optimizer,
        warmup_iters=5,
        total_iters=opt.max_epoch,
        power=1.0,
        )

    best_val_loss = float('inf')
    best_val_acc = 0.0

    for i in range(opt.max_epoch):
        
        f_start_time = time.time()

        train_sampler.set_epoch(i)
        val_sampler.set_epoch(i)

        backbone.eval()
        metric_fc.train()
        
        train_loss = 0.0
        train_corrects = 0
        train_total = 0
        start_time = time.time()

        for ii, (data_input, label, _) in tqdm(enumerate(trainloader), total=len(trainloader), disable=(rank != 0)):
            backbone.eval()
            metric_fc.train()
            data_input = data_input.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True).long()

            with torch.amp.autocast(device_type='cuda'):
                feature = backbone(data_input)
                output = metric_fc(feature, label)
                loss = criterion(output, label)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            preds = torch.argmax(output.data, dim=1)
            train_total += label.size(0)
            train_corrects += (preds == label).sum().item()
            train_loss += loss.item() * data_input.size(0)

            if (ii + 1) % opt.print_freq == 0:

                reduced_loss = loss.detach().clone()
                dist.all_reduce(reduced_loss, op=dist.ReduceOp.SUM)
                

                preds_for_log = torch.argmax(output.data, dim=1)
                corrects_for_log = (preds_for_log == label).sum()
                dist.all_reduce(corrects_for_log, op=dist.ReduceOp.SUM)

                num_samples_for_log = torch.tensor(label.size(0), device=device)
                dist.all_reduce(num_samples_for_log, op=dist.ReduceOp.SUM)
                
                if rank == 0:
                    avg_loss = reduced_loss / world_size
                    avg_acc = corrects_for_log / num_samples_for_log if num_samples_for_log > 0 else torch.tensor(0.0)
                    
                    elapsed = time.time() - start_time
                    speed = (ii + 1) / elapsed
                    current_lr = scheduler.get_last_lr()[0]

                    print(f'Epoch {i}/{opt.max_epoch-1}, Iter {ii+1}/{len(trainloader)} | '
                          f'Train Loss: {avg_loss.item():.4f}, Acc: {avg_acc.item():.4f}, Speed: {speed:.2f} iters/s lr : {current_lr:.6f}')
                    
                    if writer:
                        global_step = i * len(trainloader) + ii
                        writer.add_scalar('train/running_loss', avg_loss.item(), global_step)
                        writer.add_scalar('train/running_acc', avg_acc.item(), global_step)
                        writer.add_scalar('train/learning_rate', current_lr, global_step)

        train_total_tensor = torch.tensor(train_total, device=device)
        train_corrects_tensor = torch.tensor(train_corrects, device=device)
        train_loss_tensor = torch.tensor(train_loss, device=device)

        dist.all_reduce(train_total_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(train_corrects_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)

        if rank == 0:
            if train_total_tensor.item() > 0:
                epoch_train_loss = train_loss_tensor.item() / train_total_tensor.item()
                epoch_train_acc = train_corrects_tensor.item() / train_total_tensor.item()

            else:
                epoch_train_loss = 0.0
                epoch_train_acc = 0.0

            if writer:
                writer.add_scalar('train/epoch_loss', epoch_train_loss, i)
                writer.add_scalar('train/epoch_acc', epoch_train_acc, i)
            
            f_end_time = time.time()
            print(f'--- Epoch {i} Train Summary --- Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc:.4f} Total time  : {f_end_time - f_start_time}')
            
            save_model(backbone , metric_fc , opt.checkpoints_path , opt.backbone , i) # 훈련마다 저장
        
        
        scheduler.step()


        if rank == 0:
            os.makedirs(opt.checkpoints_path, exist_ok=True)
            save_model(backbone, metric_fc, opt.checkpoints_path, opt.backbone, i)

        dist.barrier() # rank0저장 완료까지 서브프로세스 대기

        backbone.eval()
        metric_fc.eval()
        
        val_loss = 0.0
        val_corrects = 0
        val_total = 0

        with torch.no_grad():
            for data_input, label, _ in tqdm(valloader, desc=f"Epoch {i}/{opt.max_epoch-1} [Val]", disable=(rank != 0) , total=len(valloader)):
                data_input = data_input.to(device, non_blocking=True)
                label = label.to(device, non_blocking=True).long()

                feature = backbone(data_input)
                output = metric_fc(feature, label)
                loss = criterion(output, label)

                preds = torch.argmax(output.data, dim=1)
                val_total += label.size(0)
                val_corrects += (preds == label).sum().item()
                val_loss += loss.item() * data_input.size(0)

        val_total_tensor = torch.tensor(val_total, device=device)
        val_corrects_tensor = torch.tensor(val_corrects, device=device)
        val_loss_tensor = torch.tensor(val_loss, device=device)

        dist.all_reduce(val_total_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_corrects_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)

        if rank == 0:

            if val_total_tensor.item() > 0:
                epoch_val_loss = val_loss_tensor.item() / val_total_tensor.item()
                epoch_val_acc = val_corrects_tensor.item() / val_total_tensor.item()
            else:
                epoch_val_loss = 0.0
                epoch_val_acc = 0.0

            if writer:
                writer.add_scalar('val/epoch_loss', epoch_val_loss, i)
                writer.add_scalar('val/epoch_acc', epoch_val_acc, i)

            print(f'--- Epoch {i} Validation Summary --- Loss: {epoch_val_loss:.4f}, Acc: {epoch_val_acc:.4f}')

            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_val_acc = epoch_val_acc

                if rank == 0:
                    logging.info(f"New best model at epoch {i} with loss: {best_val_loss:.4f}, acc: {best_val_acc:.4f}")
                    
                    best_save_dir = os.path.join('checkpints_moduler')
                    os.makedirs(best_save_dir, exist_ok=True)


                    best_backbone_path = os.path.join(best_save_dir, f'{opt.backbone}_backbone_best_{i}.pth')
                    best_head_path = os.path.join(best_save_dir, f'{opt.backbone}_head_best.pth_{i}')
              
                    checkpoint = {
                        'epoch': i,
                        'backbone_state_dict': backbone.module.state_dict() if hasattr(backbone, 'module') else backbone.state_dict(),
                        'head_state_dict': metric_fc.module.state_dict() if hasattr(metric_fc, 'module') else metric_fc.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'best_val_loss': best_val_loss,
                    }

                    torch.save(
                    backbone.module.state_dict() if hasattr(backbone, 'module') else backbone.state_dict(),
                    best_backbone_path
                         )
                    torch.save(
                        metric_fc.module.state_dict() if hasattr(metric_fc, 'module') else metric_fc.state_dict(),
                    best_head_path
                    )

                    best_checkpoint_path = os.path.join(best_save_dir, f'{opt.backbone}_checkpoint_best.pth')

                    torch.save(checkpoint, best_checkpoint_path)

        dist.barrier()

    if writer:
        writer.close()
    cleanup_ddp()

if __name__ == '__main__':
    main()