import argparse
import os

class Config:
    def __init__(self):
        self.backbone = 'irsnet50'
        self.metric = 'arc_margin'
        self.easy_margin = True
        self.loss = 'cross_entropy'

        self.batch_size = 256 # 256 , 384 ,448 , 512
        self.input_size = (3, 112, 112)
        self.max_epoch = 100
        self.backbone_lr = 5e-3 # 사전학습 1e-4 or 5e-4 처음 1e-3 , 5e-3
        self.head_lr = 1e-3
        self.lr_step = 10
        self.lr_decay = 0.95
        self.weight_decay = 1e-4 # 데이터셋이 작다면 1e-4, 데이터셋이 크다면 5e-4
        self.optimizer = 'adamw'

        self.train_root = './pair_aligned' 

        self.checkpoint = 'checkpoint'
        self.print_freq = 50
        self.save_interval = 10


        self.backbone_pretrained_weights = 'models/weight/backbone_ir50_asia.pth'
        self.head_pretrained_weights = None

        self.optimizer_pretrained = None

        self.name = None

def create_parser(config):
    parser = argparse.ArgumentParser(
        description='PyTorch Training Configuration',
        # HelpFormatter to show default values from Config class
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    model_group = parser.add_argument_group('Model Architecture')

    model_group.add_argument('--backbone', type=str,
                           choices=['iresnet50', 'iresnet100'],
                           help='Backbone architecture. Default: %(default)s')
    
    model_group.add_argument('--metric', type=str,
                           choices=['add_margin', 'arc_margin', 'sphere'],
                           help='Metric learning method. Default: %(default)s')

    # For boolean flags, we add a --no- variant to be able to disable it.
    model_group.add_argument('--easy_margin', dest='easy_margin', action='store_true',
                           help='Enable easy margin for ArcFace. (default)')
    model_group.add_argument('--no_easy_margin', dest='easy_margin', action='store_false',
                           help='Disable easy margin for ArcFace.')
    
    model_group.add_argument('--loss', type=str,
                           choices=['focal_loss', 'cross_entropy'],
                           help='Loss function. Default: %(default)s')
    
    model_group.add_argument('--backbone_pretrained_weights', type=str, 
                           help='Path to pre-trained weights file. Default: %(default)s')
    
    model_group.add_argument('--head_pretrained_weights',type=str,
                           help='Path to pre-trained head weights file (optional). Default: %(default)s')
    
    # Training parameters
    train_group = parser.add_argument_group('Training Parameters')

    train_group.add_argument('--optimizer_pretrained_path',type=str,
                             help='pretrained optimzer call path. Default: %(default)s')

    train_group.add_argument('--batch_size', type=int,
                           help='Training batch size. Default: %(default)s')
    
    train_group.add_argument('--input_size', type=str,
                           help='Input shape as "C,H,W". Default: %(default)s')
    
    train_group.add_argument('--max_epoch', type=int,
                           help='Maximum number of epochs. Default: %(default)s')
    
    # lr is not in Config, so we can't set a default for it from there.
    # The user might want to set backbone_lr and head_lr.
    # The original code has --lr, but doesn't seem to use it to set backbone_lr or head_lr.
    # I will keep it as it is.
    train_group.add_argument('--lr', type=float,
                           help='Initial learning rate')
    
    train_group.add_argument('--lr_step', type=int,
                           help='Learning rate decay step. Default: %(default)s')
    
    train_group.add_argument('--lr_decay', type=float,
                           help='Learning rate decay factor. Default: %(default)s')
    
    train_group.add_argument('--weight_decay', type=float,
                           help='Weight decay. Default: %(default)s')
    
    train_group.add_argument('--optimizer', type=str,
                           choices=['sgd', 'adam' ,'adamw'],
                           help='Optimizer type. Default: %(default)s')
    
    train_group.add_argument('--print_freq', type=int,
                           help='Print frequency. Default: %(default)s')
    
    # Model paths
    model_path_group = parser.add_argument_group('Model Paths')
    model_path_group.add_argument('--checkpoint', type=str,
                                help='Checkpoints save directory. Default: %(default)s')
    model_path_group.add_argument('--save_interval', type=int,
                                help='Model save interval (epochs). Default: %(default)s')
    model_path_group.add_argument('--train_root', type=str,
                                help='Train root directory. Default: %(default)s')

    parser.set_defaults(**vars(config))
    return parser

def parse_config():
    cfg = Config()
    parser = create_parser(cfg)
    args = parser.parse_args()

    # Update the config with any arguments provided by the user
    for key, value in vars(args).items():
        if hasattr(cfg, key):
            # Special handling for input_size
            if key == 'input_size' and isinstance(value, str):
                value = tuple(map(int, value.split(',')))
            setattr(cfg, key, value)
            
    os.makedirs(cfg.checkpoint, exist_ok=True)
    
    return cfg

def get_config(rank):
    cfg = parse_config()
    if rank ==0:
        print("=" * 50)
        print("Configuration Settings")
        print("=" * 50)
        for key, value in vars(cfg).items():
            print(f"{key:<20}: {value}")
        print("=" * 50)
    return cfg

if __name__ == "__main__":
    cfg = get_config(rank=0)