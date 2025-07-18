from __future__ import print_function

import os
import time
import logging
import torch
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import config.config as config

from data.dataset import Dataset
from models import *
from models.backbone.irsnet import iresnet50, iresnet100
import utils

# --- Helper Functions ---

def save_model(backbone, metric_fc, save_path, name, iter_cnt, is_best=False):
    """Saves the model state for single GPU training."""
    os.makedirs(save_path, exist_ok=True)
    if is_best:
        save_name = os.path.join(save_path, f'{name}_best.pth')
    else:
        save_name = os.path.join(save_path, f'{name}_{iter_cnt}.pth')

    torch.save({
        'backbone_state_dict': backbone.state_dict(),
        'metric_fc_state_dict': metric_fc.state_dict(),
    }, save_name)
    logging.info(f"Model saved to {save_name}")
    return save_name

def load_weights(model, weight_path, model_name, device):
    """Loads weights for a model in single GPU context."""
    if weight_path and os.path.exists(weight_path):
        try:
            load_result = model.load_state_dict(
                torch.load(weight_path, map_location=device),
                strict=False
            )
            logging.info(f"Pre-trained weights for {model_name} loaded from {weight_path}.")
            if load_result.missing_keys or load_result.unexpected_keys:
                logging.info(f"Missing keys for {model_name}: {load_result.missing_keys}")
                logging.info(f"Unexpected keys for {model_name}: {load_result.unexpected_keys}")
        except Exception as e:
            logging.error(f"Error loading weights for {model_name}: {e}")
    else:
        logging.info(f"No pre-trained weights for {model_name} found or specified, starting from scratch.")

# --- Main Function ---

def main():
    # --- Initial Setup ---
    torch.backends.cudnn.benchmark = True
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    opt = config.get_config()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter()

    logging.info(f"Using device: {device}")
    if not torch.cuda.is_available():
        logging.warning("CUDA is not available. Training on CPU.")

    # --- Dataset and DataLoader ---
    full_dataset = Dataset(root=opt.train_root, phase='train', input_shape=(1, 112, 112))
    train_size = int(0.7 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    trainloader = data.DataLoader(
        train_dataset, batch_size=opt.train_batch_size, shuffle=True,
        num_workers=opt.num_workers, pin_memory=True
    )
    valloader = data.DataLoader(
        val_dataset, batch_size=opt.train_batch_size, shuffle=False,
        num_workers=opt.num_workers, pin_memory=True
    )
    logging.info(f"Training set size: {len(train_dataset)}")
    logging.info(f"Validation set size: {len(val_dataset)}")

    # --- Model, Loss, Optimizer ---
    if opt.loss == 'focal_loss':
        criterion = FocalLoss(gamma=2)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    if opt.backbone == 'iresnet50':
        backbone = iresnet50()
        logging.info("Using IResNet-50 backbone")
    else: # Default or iresnet100
        backbone = iresnet100()
        logging.info("Using IResNet-100 backbone")

    num_classes = full_dataset.get_classes
    if opt.metric == 'add_margin':
        metric_fc = AddMarginProduct(512, num_classes, s=30, m=0.35)
    elif opt.metric == 'arc_margin':
        metric_fc = ArcMarginProduct(512, num_classes, s=30, m=0.5, easy_margin=opt.easy_margin)
    elif opt.metric == 'sphere':
        metric_fc = SphereProduct(512, num_classes, m=4)
    else:
        metric_fc = ArcMarginProduct(512, num_classes, s=30, m=0.5, easy_margin=opt.easy_margin)

    # Load weights using the new helper function
    load_weights(backbone, getattr(opt, 'backbone_pretrained_weights', None), "Backbone", device)
    load_weights(metric_fc, getattr(opt, 'head_pretrained_weight', None), "Head", device)

    # Freeze backbone parameters
    for param in backbone.parameters():
        param.requires_grad = False
    
    backbone.to(device)
    metric_fc.to(device)

    if opt.optimizer == 'sgd':
        optimizer = torch.optim.SGD(metric_fc.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    elif opt.optimizer == 'adam':
        optimizer = torch.optim.Adam(metric_fc.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    else:  # adamw
        optimizer = torch.optim.AdamW(metric_fc.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

    scheduler = utils.lr_scheduler.PolynomialLRWarmup(
        optimizer, warmup_iters=5, total_iters=opt.max_epoch, power=1.0
    )
    scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())

    # --- Training & Validation Loop ---
    best_val_loss = float('inf')
    best_val_acc = 0.0

    for epoch in range(opt.max_epoch):
        # --- Training Phase ---
        backbone.eval()
        metric_fc.train()
        train_loss, train_corrects, train_total = 0.0, 0, 0
        
        pbar = tqdm(enumerate(trainloader), total=len(trainloader), desc=f"Epoch {epoch}/{opt.max_epoch-1} [Train]")
        for ii, (data_input, label, _) in pbar:
            data_input = data_input.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True).long()

            with torch.amp.autocast(device_type='cuda', enabled=torch.cuda.is_available()):
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

            # Detailed logging
            if (ii + 1) % opt.print_freq == 0:
                avg_loss = train_loss / train_total if train_total > 0 else 0
                avg_acc = train_corrects / train_total if train_total > 0 else 0
                current_lr = scheduler.get_last_lr()[0]
                
                logging.info(f'Epoch {epoch}/{opt.max_epoch-1}, Iter {ii+1}/{len(trainloader)} | '
                             f'Train Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}, LR: {current_lr:.6f}')
                
                global_step = epoch * len(trainloader) + ii
                writer.add_scalar('train/running_loss', avg_loss, global_step)
                writer.add_scalar('train/running_acc', avg_acc, global_step)
                writer.add_scalar('train/learning_rate', current_lr, global_step)

        epoch_train_loss = train_loss / train_total if train_total > 0 else 0
        epoch_train_acc = train_corrects / train_total if train_total > 0 else 0
        writer.add_scalar('train/epoch_loss', epoch_train_loss, epoch)
        writer.add_scalar('train/epoch_acc', epoch_train_acc, epoch)
        logging.info(f'--- Epoch {epoch} Train Summary --- Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc:.4f}')

        scheduler.step()

        # --- Validation Phase ---
        backbone.eval()
        metric_fc.eval()
        val_loss, val_corrects, val_total = 0.0, 0, 0
        
        with torch.no_grad():
            pbar_val = tqdm(valloader, desc=f"Epoch {epoch}/{opt.max_epoch-1} [Val]")
            for data_input, label, _ in pbar_val:
                data_input = data_input.to(device, non_blocking=True)
                label = label.to(device, non_blocking=True).long()

                with torch.amp.autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                    feature = backbone(data_input)
                    output = metric_fc(feature, label)
                    loss = criterion(output, label)

                preds = torch.argmax(output.data, dim=1)
                val_total += label.size(0)
                val_corrects += (preds == label).sum().item()
                val_loss += loss.item() * data_input.size(0)
                
                pbar_val.set_postfix({
                    'Loss': f'{val_loss/val_total:.4f}' if val_total > 0 else '0.0',
                    'Acc': f'{val_corrects/val_total:.4f}' if val_total > 0 else '0.0'
                })

        epoch_val_loss = val_loss / val_total if val_total > 0 else 0
        epoch_val_acc = val_corrects / val_total if val_total > 0 else 0
        writer.add_scalar('val/epoch_loss', epoch_val_loss, epoch)
        writer.add_scalar('val/epoch_acc', epoch_val_acc, epoch)
        logging.info(f'--- Epoch {epoch} Validation Summary --- Loss: {epoch_val_loss:.4f}, Acc: {epoch_val_acc:.4f}')

        # --- Save Model ---
        save_model(backbone, metric_fc, opt.checkpoints_path, opt.backbone, epoch)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_val_acc = epoch_val_acc
            logging.info(f"New best model at epoch {epoch} with loss: {best_val_loss:.4f}, acc: {best_val_acc:.4f}")
            best_save_dir = os.path.join(opt.checkpoints_path, 'best')
            save_model(backbone, metric_fc, best_save_dir, opt.backbone, epoch, is_best=True)


    writer.close()
    logging.info("Training finished.")

if __name__ == '__main__':
    main()
