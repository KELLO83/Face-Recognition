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
import torch.multiprocessing as mp
import logging
from tqdm import tqdm

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

    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node,))

def main_worker(rank, world_size):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(f"GPU Available: {torch.cuda.device_count()}")
    device_id = setup_ddp(rank, world_size)

    logging.info(f"Process {rank} started on device {device_id}")

    writer = SummaryWriter(log_dir='./logs/train') if rank == 0 else None
    opt = config.get_config(rank)


    scaler = torch.amp.GradScaler(device=f'cuda:{device_id}')
    device = torch.device(f"cuda:{device_id}")

    full_dataset = Dataset(root=f'{opt.train_root}', phase='train', input_shape=(1, 112, 112))

    train_size = int(0.7 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])


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


    if opt.loss == 'focal_loss':
        criterion = FocalLoss(gamma=2)
    else:
        criterion = torch.nn.CrossEntropyLoss()

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

    backbone_params = sum(p.numel() for p in backbone.parameters())
    trainable_backbone_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)

    if rank == 0:
        logging.info('==' * 30)
        logging.info(f"Backbone total params: {backbone_params:,}")
        logging.info(f"Backbone trainable params: {trainable_backbone_params:,}")
        logging.info(f"Head total params: {sum(p.numel() for p in metric_fc.parameters()):,} ")
        logging.info(f"Head trainable params: {sum(p.numel() for p in metric_fc.parameters() if p.requires_grad):,}")
        logging.info('==' * 30)

    num_classes = full_dataset.get_classes
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

    backbone.to(device)
    metric_fc.to(device)

    #backbone = DistributedDataParallel(backbone, device_ids=[device_id], find_unused_parameters=False)
    metric_fc = DistributedDataParallel(metric_fc, device_ids=[device_id], find_unused_parameters=False)

    backbone = torch.compile(backbone, options={"max_autotune": False})



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

            if rank == 0 and (ii + 1) % opt.print_freq == 0:
                elapsed = time.time() - start_time
                speed = (ii + 1) / elapsed
                avg_loss = train_loss / train_total if train_total > 0 else 0
                avg_acc = train_corrects / train_total if train_total > 0 else 0
                current_lr = scheduler.get_last_lr()[0]

                print(f'Epoch {i}/{opt.max_epoch-1}, Iter {ii+1}/{len(trainloader)} | '
                      f'Train Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}, Speed: {speed:.2f} iters/s lr : {current_lr:.6f}')
                
                if writer:
                    global_step = i * len(trainloader) + ii
                    writer.add_scalar('train/running_loss', avg_loss, global_step)
                    writer.add_scalar('train/running_acc', avg_acc, global_step)
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
            
            print(f'--- Epoch {i} Train Summary --- Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc:.4f}')

        scheduler.step()

        backbone.eval()
        metric_fc.eval()
        
        val_loss = 0.0
        val_corrects = 0
        val_total = 0

        with torch.no_grad():
            for data_input, label, _ in valloader:
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


        if rank == 0:
            save_model(backbone, metric_fc, opt.checkpoints_path, opt.backbone, i)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_val_acc = epoch_val_acc

            if rank == 0:
                best_save_dir = os.path.join(opt.checkpoints_path, 'best')
                os.makedirs(best_save_dir, exist_ok=True)
                best_save_path = os.path.join(best_save_dir, f'{opt.backbone}_best.pth')
                logging.info(f"New best model at epoch {i} with loss: {best_val_loss:.4f}, acc: {best_val_acc:.4f}")
                torch.save({
                    'backbone_state_dict': backbone.module.state_dict() if hasattr(backbone, 'module') else backbone.state_dict(),
                    'metric_fc_state_dict': metric_fc.module.state_dict() if hasattr(metric_fc, 'module') else metric_fc.state_dict(),
                }, best_save_path)

        dist.barrier()

    if writer:
        writer.close()
    cleanup_ddp()

if __name__ == '__main__':
    main()