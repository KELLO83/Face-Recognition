import argparse
import os

class Config:
    def __init__(self):
        self.backbone = 'irsnet50'
        self.metric = 'arc_margin'
        self.easy_margin = True
        self.loss = 'cross_entropy'

        self.train_batch_size = 896 # 256 , 384 ,448 , 512
        self.input_size = (3, 112, 112)
        self.max_epoch = 300
        self.backbone_lr = 1e-4
        self.head_lr = 1e-3
        self.lr_step = 10
        self.lr_decay = 0.95
        self.weight_decay = 5e-4
        self.optimizer = 'adamw'

        self.train_root = './pair' 

        self.checkpoints_path = 'checkpoints'
        self.print_freq = 50
        self.save_interval = 10


        self.backbone_pretrained_weights = 'models/weight/backbone_ir50_asia.pth'
        self.head_pretrained_weights = None

        self.optimizer_pretrained = None

def create_parser():

    parser = argparse.ArgumentParser(
        description='PyTorch Training Configuration',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    model_group = parser.add_argument_group('Model Architecture')

    model_group.add_argument('--backbone', type=str, default=None,
                           choices=['iresnet50', 'iresnet100'],
                           help='Backbone architecture')
    

    model_group.add_argument('--metric', type=str, default=None,
                           choices=['add_margin', 'arc_margin', 'sphere'],
                           help='Metric learning method')

    model_group.add_argument('--easy_margin', action='store_true', default=False,
                           help='Use easy margin for ArcFace')
    
    model_group.add_argument('--loss', type=str, default='cross_entropy',
                           choices=['focal_loss', 'cross_entropy'],
                           help='Loss  function')
    

    model_group.add_argument('--backbone_pretrained_weights', type=str, 
                             default=None,
                             choices=['models/weight/ms1mv3_arcface_r50_fp16.pth',
                                      'models/weight/ms1mv3_arcface_r100_fp16.pth'],
                           help='Path to pre-trained weights file')
    
    model_group.add_argument('--head_pretrained_weights',type=str, default=None,
                           help='Path to pre-trained head weights file (optional)')
    
    # Training parameters
    train_group = parser.add_argument_group('Training Parameters')

    train_group.add_argument('--optimizer_pretrained_path',type=str,
                             help='pretrained optimzer call path')

    train_group.add_argument('--train_batch_size', type=int,
                           help='Training batch size')
    
    train_group.add_argument('--input_size', type=str, default='1,112,112',
                           help='Input shape as "C,H,W"')
    
    train_group.add_argument('--max_epoch', type=int, default=None,
                           help='Maximum number of epochs')
    
    train_group.add_argument('--lr', type=float, default=None,
                           help='Initial learning rate')
    
    train_group.add_argument('--lr_step', type=int, default=None,
                           help='Learning rate decay step')
    
    train_group.add_argument('--lr_decay', type=float, default=None,
                           help='Learning rate decay factor')
    
    train_group.add_argument('--weight_decay', type=float, default=None,
                           help='Weight decay')
    
    train_group.add_argument('--optimizer', type=str, default='adamw',
                           choices=['sgd', 'adam' ,'adamw'],
                           help='Optimizer type')
    
    train_group.add_argument('--print_freq', type=int, default=10,
                           help='Print frequency')
    

    # Model paths
    model_path_group = parser.add_argument_group('Model Paths')
    model_path_group.add_argument('--checkpoints_path', type=str, default='checkpoints',
                                help='Checkpoints save directory')
    model_path_group.add_argument('--load_model_path', type=str, default='models/resnet18.pth',
                                help='Pre-trained model path')
    model_path_group.add_argument('--test_model_path', type=str, default='checkpoints/resnet18_110.pth',
                                help='Model path for testing')
    model_path_group.add_argument('--save_interval', type=int, default=10,
                                help='Model save interval (epochs)')
    model_path_group.add_argument('--train_root', type=str,
                                help='Train root directory')
    

    return parser
    

def parse_config():
    parser = create_parser()
    args = parser.parse_args()

    cfg = Config()

    for key, value in vars(args).items():
        if hasattr(cfg, key):
            if value is not None:
                if key == 'input_size' and isinstance(value, str):
                    value = tuple(map(int, value.split(',')))
                setattr(cfg, key, value)
            
    os.makedirs(cfg.checkpoints_path, exist_ok=True)
    
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

