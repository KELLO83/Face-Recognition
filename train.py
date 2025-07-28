from __future__ import print_function

import logging
import torch
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import config.config as config
from utils.earlystop_ import EarlyStopping
from data.dataset import Dataset
from models import *
#from models.backbone.irsnet import iresnet50, iresnet100 #ms1mv3_arcface_r50_fp16
from models.backbone.ir_ASIS_Resnet import Backbone
import utils
import  wandb

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def wandb_init(Config):
    name = Config.name
    del Config.name
    wandb.init(
        project='arcface-pytorch',
        name=name,        
        config=Config,
    )



def save_model(backbone, metric_fc, save_path, name, iter_cnt, is_best=False):

    model_dir = os.path.join(save_path, name)
    os.makedirs(model_dir, exist_ok=True)

    if is_best:
        backbone_name = os.path.join(model_dir, f'{name}_best.pth')
        head_name = os.path.join(model_dir, 'head_best.pth')
    else:
        backbone_name = os.path.join(model_dir, f'{name}_{iter_cnt}.pth')
        head_name = os.path.join(model_dir, f'head_{iter_cnt}.pth')

    torch.save(backbone.state_dict(), backbone_name)

    torch.save(metric_fc.state_dict() , head_name)

    logging.info(f"Model backbone saved to {backbone_name}")
    logging.info(f"Model head saved to {head_name}")
    return backbone_name

def load_weights(model, weight_path, model_name, device):
    """Loads weights for a model in single GPU context."""
    if weight_path and os.path.exists(weight_path) and weight_path != 'None':

        try:
            print(f"Loading weights for {model_name} from {weight_path}")

            weight = torch.load(weight_path, map_location=device)
            load_result = model.load_state_dict(
                weight,
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

def train_log_file(log_path, message):
    """Appends a log message to a file."""
    with open(log_path, 'a') as f:
        f.write(message + '\n')

def main():
    rank = 0
    total_cpus = os.cpu_count() or 1
    num_workers = max(1, total_cpus // 2)

    stopping = EarlyStopping(patience=10, verbose=True, delta=1e-2)


    try:
        wandb_init(config.get_config(rank=rank))
    except Exception as e:
        logging.error(f"Error initializing wandb: {e}")
        wandb.init(mode="disabled")


    print(f"Total CPUs: {total_cpus}, Using {num_workers} workers for data loading.")
    torch.backends.cudnn.benchmark = True
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    opt = config.get_config(rank = rank)
    
    os.makedirs(opt.checkpoints_path, exist_ok=True)
    log_file_path = os.path.join(opt.checkpoints_path, "training_log.txt")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    writer = SummaryWriter()

    logging.info(f"Using device: {device}")
    if not torch.cuda.is_available():
        logging.warning("CUDA is not available. Training on CPU.")

    full_dataset = Dataset(root=opt.train_root, phase='train', input_shape=(3, 112, 112))
        
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    torch.manual_seed(42)
    train_indices, val_indices = torch.utils.data.random_split(
        range(len(full_dataset)), [train_size, val_size]
    )


    train_full_dataset = Dataset(root=opt.train_root, phase='train', input_shape=(3, 112, 112))
    val_full_dataset = Dataset(root=opt.train_root, phase='val', input_shape=(3, 112, 112))

    train_dataset = torch.utils.data.Subset(train_full_dataset, train_indices.indices)
    val_dataset = torch.utils.data.Subset(val_full_dataset, val_indices.indices)

    trainloader = data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    valloader = data.DataLoader(
        val_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    logging.info(f"Training set size: {len(train_dataset)}")
    logging.info(f"Validation set size: {len(val_dataset)}")
    logging.info("total classes: {}".format(full_dataset.get_classes))


    if opt.loss == 'focal_loss':
        criterion = FocalLoss(gamma=2)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    if opt.backbone == 'irsnet50':
        backbone= Backbone(input_size=(112, 112), num_layers=50)
        logging.info("Using Resnet-50 backbone")
    elif opt.backbone == 'irsnet100': # Default or iresnet100
        backbone = Backbone(input_size=(112, 112), num_layers=100, mode='ir_se')
        logging.info("Using ir_ASIS_Resnet-100 backbone")
    else:
        raise ValueError(f"Unsupported backbone architecture: {opt.backbone}")

    num_classes = full_dataset.get_classes

    if opt.metric == 'add_margin':
        metric_fc = AddMarginProduct(512, num_classes, s=30, m=0.35)
    elif opt.metric == 'arc_margin':
        metric_fc = ArcMarginProduct(512, num_classes, s=30, m=0.35, easy_margin=opt.easy_margin)
    elif opt.metric == 'sphere':
        metric_fc = SphereProduct(512, num_classes, m=4)
    else:
        metric_fc = ArcMarginProduct(512, num_classes, s=30, m=0.35, easy_margin=opt.easy_margin)

    if opt.backbone_pretrained_weights != None and opt.backbone == 'irsnet50':
        load_weights(backbone, opt.backbone_pretrained_weights,opt.backbone,device)
    if opt.head_pretrained_weights != None:
        load_weights(backbone, opt.head_pretrained_weights, opt.metric, device)



    #백본 일부 동결 
    if opt.backbone == 'irsnet50':
        for name, param in backbone.named_parameters():
            if 'body.' in name:
                try:
                    body_idx = int(name.split('.')[1])
                    if body_idx < 23:
                        param.requires_grad = False
                except (ValueError, IndexError):
                    continue
            elif 'input_layer' in name:
                param.requires_grad = False

    
    try:
        wandb.watch((backbone, metric_fc), log='all', log_freq=opt.print_freq)

    except Exception as e:
        logging.error(f"Error setting up wandb watch: {e}")

    backbone.to(device)
    metric_fc.to(device)

    if opt.optimizer == 'adam':
        optimizer = torch.optim.Adam([
            {'params': [p for p in backbone.parameters() if p.requires_grad], 'lr': opt.backbone_lr}, 
            {'params': metric_fc.parameters(), 'lr': opt.head_lr}
        ], weight_decay=opt.weight_decay)

    elif opt.optimizer == 'adamw':
        optimizer = torch.optim.AdamW([
            {'params': [p for p in backbone.parameters() if p.requires_grad], 'lr': opt.backbone_lr}, 
            {'params': metric_fc.parameters(), 'lr': opt.head_lr}
        ], weight_decay=opt.weight_decay)

    else: 
        optimizer = torch.optim.AdamW([
            {'params': [p for p in backbone.parameters() if p.requires_grad], 'lr': opt.backbone_lr}, 
            {'params': metric_fc.parameters(), 'lr': opt.head_lr}
        ], weight_decay=opt.weight_decay)

    if rank == 0:
        backbone_params = sum(p.numel() for p in backbone.parameters()) 
        trainable_backbone_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
        trainable_percentage = (trainable_backbone_params / backbone_params) * 100 if backbone_params > 0 else 0

        logging.info('==' * 30)
        logging.info(f"Backbone total params: {backbone_params:,}")
        logging.info(f"Backbone trainable params: {trainable_backbone_params:,}")
        logging.info(f"Head total params: {sum(p.numel() for p in metric_fc.parameters()):,} ")
        logging.info(f"Trainable Backbone Percentage: {trainable_percentage:.2f}%")
        logging.info(f"Head trainable params: {sum(p.numel() for p in metric_fc.parameters() if p.requires_grad):,}")
        logging.info('==' * 30)

    interactive_mode = os.getenv('INTERACTIVE_MODE', 'true').lower() == 'true'
    
    if interactive_mode:
        logging.info("훈련을 시작하려면 아무 키나 입력하세요... 종료 (1)")
        running = input("")
        if running == '1' :
            logging.info("훈련을 종료합니다.")
            return
        
    scheduler = utils.lr_scheduler.PolynomialLRWarmup(
        optimizer, warmup_iters=10, total_iters=opt.max_epoch, power=1.0 , limit_lr = 1e-5
    )
    scaler = torch.amp.GradScaler(enabled=False)

    best_val_loss = float('inf')
    best_val_acc = 0.0

    for epoch in range(opt.max_epoch):

        backbone.train()
        metric_fc.train()
        train_loss, train_corrects, train_total = 0.0, 0, 0
        
        pbar = tqdm(enumerate(trainloader), total=len(trainloader), desc=f"Epoch {epoch}/{opt.max_epoch-1} [Train]")
        for ii, (data_input, label, _) in pbar:
            data_input = data_input.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True).long()

            with torch.amp.autocast(device_type='cuda', enabled=False):
                feature = backbone(data_input)
                output = metric_fc(feature, label)
                loss = criterion(output, label)

            optimizer.zero_grad()
            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(backbone.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(metric_fc.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()

            preds = torch.argmax(output.data, dim=1)
            train_total += label.size(0)
            train_corrects += (preds == label).sum().item()
            train_loss += loss.item() * data_input.size(0)

            if (ii + 1) % opt.print_freq == 0:
                avg_loss = train_loss / train_total if train_total > 0 else 0
                avg_acc = train_corrects / train_total if train_total > 0 else 0
                backbone_lr = scheduler.get_last_lr()[0]
                head_lr = scheduler.get_last_lr()[1] 

                log_message = (f'Epoch {epoch}/{opt.max_epoch-1}, Iter {ii+1}/{len(trainloader)} | '
                               f'Train Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}')
                logging.info(log_message)
                log_message += f'Backbone LR: {backbone_lr:.6f}, Head LR: {head_lr:.6f}'
                train_log_file(log_file_path, log_message)
                
                global_step = epoch * len(trainloader) + ii
                writer.add_scalar('train/running_loss', avg_loss, global_step)
                writer.add_scalar('train/running_acc', avg_acc, global_step)
                writer.add_scalar('train/backbone_learning_rate', backbone_lr, global_step)

                try:
                    wandb.log({
                        'train_loss': avg_loss,
                        'train_acc': avg_acc,
                        'backbone_learning_rate': backbone_lr,
                        'head_learning_rate': head_lr,
                        'epoch': epoch,
                    })
                except Exception as e:
                    logging.error(f"Error logging to wandb: {e}")

        epoch_train_loss = train_loss / train_total if train_total > 0 else 0
        epoch_train_acc = train_corrects / train_total if train_total > 0 else 0
        writer.add_scalar('train/epoch_loss', epoch_train_loss, epoch)
        writer.add_scalar('train/epoch_acc', epoch_train_acc, epoch)
        
        train_summary_message = f'--- Epoch {epoch} Train Summary --- Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc:.4f}'
        logging.info(train_summary_message)
        train_log_file(log_file_path, train_summary_message)


        backbone.eval()
        metric_fc.eval()
        val_loss, val_corrects, val_total = 0.0, 0, 0
        
        with torch.no_grad():
            pbar_val = tqdm(valloader, desc=f"Epoch {epoch}/{opt.max_epoch-1} [Val]")
            for data_input, label, _ in pbar_val:
                data_input = data_input.to(device, non_blocking=True)
                label = label.to(device, non_blocking=True).long()

                with torch.amp.autocast(device_type='cuda', enabled=False):
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

        scheduler.step()
        epoch_val_loss = val_loss / val_total if val_total > 0 else 0
        epoch_val_acc = val_corrects / val_total if val_total > 0 else 0
        writer.add_scalar('val/epoch_loss', epoch_val_loss, epoch)
        writer.add_scalar('val/epoch_acc', epoch_val_acc, epoch)
        try:
            wandb.log({
                'val_loss': epoch_val_loss,
                'val_acc': epoch_val_acc,
                'epoch': epoch
            })
        except Exception as e: # 조기 종료를 의미하며 초기값은 False로 설정
            logging.error(f"Error logging to wandb: {e}")

        val_summary_message = f'--- Epoch {epoch} Validation Summary --- Loss: {epoch_val_loss:.4f}, Acc: {epoch_val_acc:.4f}'
        logging.info(val_summary_message)
        train_log_file(log_file_path, val_summary_message)


        should_save_regular = (opt.save_interval > 0 and epoch % opt.save_interval == 0)
        should_save_first = (epoch == 0)  
        if should_save_first or should_save_regular:
            logging.info(f"Saving model at epoch {epoch} to {opt.checkpoints_path}")
            save_model(backbone, metric_fc, opt.checkpoints_path,opt.backbone, epoch)



        early_stop_flag = stopping(epoch_val_loss)
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_val_acc = epoch_val_acc
            
            best_model_message = f"New best model at epoch {epoch} with loss: {best_val_loss:.4f}, acc: {best_val_acc:.4f}"
            logging.info(best_model_message)
            train_log_file(log_file_path, best_model_message)

            best_save_dir = os.path.join(opt.checkpoints_path, 'best')
            save_model(backbone, metric_fc, best_save_dir, opt.backbone, epoch, is_best=True)

        
        if early_stop_flag:
            early_stop_message = f"Early stopping triggered at epoch {epoch}. Training stopped."
            logging.info(early_stop_message)
            train_log_file(log_file_path, early_stop_message)

            final_message = f"Training completed early. Best validation - Loss: {best_val_loss:.4f}, Acc: {best_val_acc:.4f}"
            logging.info(final_message)
            train_log_file(log_file_path, final_message)
            
            try:
                wandb.log({
                    'early_stop_epoch': epoch,
                    'final_best_val_loss': best_val_loss,
                    'final_best_val_acc': best_val_acc,
                    'training_completed': True
                })
            except Exception as e:
                logging.error(f"Error logging early stop to wandb: {e}")

            break

    writer.close()
    logging.info("Training finished.")

if __name__ == '__main__':
    main()

