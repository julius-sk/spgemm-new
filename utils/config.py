"""
MaxK-GNN Training Configuration
Fixed to properly inherit from ArgumentParser
"""
import argparse
import os
import time
from pathlib import Path

class TrainConfig(argparse.ArgumentParser):
    """
    Training configuration with ArgumentParser support
    Fixed to include parse_known_args() method
    """
    
    def __init__(self):
        super().__init__(description='MaxK-GNN Training Configuration')
        
        # Dataset configuration
        self.add_argument('--dataset', type=str, default='reddit',
                         choices=['reddit', 'flickr', 'yelp', 'ogbn-arxiv', 'ogbn-products', 'ogbn-proteins'],
                         help='Dataset to use for training')
        
        self.add_argument('--data_path', type=str, default='./data/',
                         help='Path to dataset directory')
        
        # Model configuration
        self.add_argument('--model', type=str, default='sage',
                         choices=['sage', 'gcn', 'gin', 'gnn_res'],
                         help='Model architecture to use')
        
        self.add_argument('--hidden_dim', type=int, default=256,
                         help='Hidden layer dimension')
        
        self.add_argument('--hidden_layers', type=int, default=3,
                         help='Number of hidden layers')
        
        self.add_argument('--dropout', type=float, default=0.5,
                         help='Dropout rate')
        
        self.add_argument('--norm', action='store_true', default=False,
                         help='Enable layer normalization')
        
        # MaxK configuration
        self.add_argument('--nonlinear', type=str, default='maxk',
                         choices=['maxk', 'relu'],
                         help='Nonlinearity function to use')
        
        self.add_argument('--maxk', type=int, default=32,
                         help='MaxK parameter (number of top elements to keep)')
        
        # Training configuration
        self.add_argument('--epochs', type=int, default=1000,
                         help='Number of training epochs')
        
        self.add_argument('--w_lr', type=float, default=0.01,
                         help='Learning rate')
        
        self.add_argument('--w_weight_decay', type=float, default=0.0,
                         help='Weight decay (L2 regularization)')
        
        self.add_argument('--enable_lookahead', action='store_true', default=False,
                         help='Enable Lookahead optimizer')
        
        # Hardware configuration
        self.add_argument('--gpu', type=int, default=0,
                         help='GPU device ID')
        
        self.add_argument('--seed', type=int, default=97,
                         help='Random seed for reproducibility')
        
        # Experimental configuration
        self.add_argument('--selfloop', action='store_true', default=False,
                         help='Add self-loops to graph')
        
        # Output configuration
        self.add_argument('--path', type=str, default=None,
                         help='Output directory for results')
        
        self.add_argument('--plot_path', type=str, default=None,
                         help='Directory for plots')
        
        self.add_argument('--evaluate', type=str, default=None,
                         help='Evaluation mode')
        
        # Performance monitoring
        self.add_argument('--log_every', type=int, default=100,
                         help='Log frequency (epochs)')
        
        self.add_argument('--eval_every', type=int, default=100,
                         help='Evaluation frequency (epochs)')
        
        self.add_argument('--save_every', type=int, default=500,
                         help='Model save frequency (epochs)')
        
        # Advanced options
        self.add_argument('--timing', action='store_true', default=False,
                         help='Enable detailed timing measurements')
        
        self.add_argument('--profile', action='store_true', default=False,
                         help='Enable profiling mode')
        
        self.add_argument('--debug', action='store_true', default=False,
                         help='Enable debug mode')
    
    def parse_args(self, args=None, namespace=None):
        """Parse arguments and set up derived paths"""
        config = super().parse_args(args, namespace)
        
        # Set up output paths if not specified
        if config.path is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            config.path = f"experiments/{config.dataset}_{config.model}_maxk{config.maxk}_{timestamp}"
        
        if config.plot_path is None:
            config.plot_path = os.path.join(config.path, "plots")
        
        # Create directories
        Path(config.path).mkdir(parents=True, exist_ok=True)
        Path(config.plot_path).mkdir(parents=True, exist_ok=True)
        
        # Set up data path
        Path(config.data_path).mkdir(parents=True, exist_ok=True)
        
        return config
    def print_params(self, logger_func=print):
        """Print parameters using a logger function"""
        # Get the current parsed arguments
        if hasattr(self, '_parsed_args'):
            config_dict = vars(self._parsed_args)
        else:
            try:
                config_dict = vars(self.parse_known_args()[0])
            except:
                # Fallback to default values
                config_dict = {
                    'dataset': getattr(self, 'dataset', 'reddit'),
                    'model': getattr(self, 'model', 'sage'),
                    'maxk': getattr(self, 'maxk', 32),
                    'hidden_dim': getattr(self, 'hidden_dim', 256),
                    'hidden_layers': getattr(self, 'hidden_layers', 3),
                    'dropout': getattr(self, 'dropout', 0.5),
                    'epochs': getattr(self, 'epochs', 1000),
                    'w_lr': getattr(self, 'w_lr', 0.01),
                    'gpu': getattr(self, 'gpu', 0),
                    'seed': getattr(self, 'seed', 97)
                }
        
        logger_func("\n🔧 Training Parameters:")
        logger_func("=" * 25)
        for key, value in sorted(config_dict.items()):
            logger_func(f"{key:20s}: {value}")
        logger_func("=" * 25)
        
    def parse_known_args(self, args=None, namespace=None):
        """Parse known arguments - this is what was missing!"""
        config, unknown = super().parse_known_args(args, namespace)
        
        # Set up output paths if not specified
        if config.path is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            config.path = f"experiments/{config.dataset}_{config.model}_maxk{config.maxk}_{timestamp}"
        
        if config.plot_path is None:
            config.plot_path = os.path.join(config.path, "plots")
        
        # Create directories
        Path(config.path).mkdir(parents=True, exist_ok=True)
        Path(config.plot_path).mkdir(parents=True, exist_ok=True)
        
        # Set up data path
        Path(config.data_path).mkdir(parents=True, exist_ok=True)
        
        return config, unknown
    
    def save_config(self, path):
        """Save configuration to file"""
        config_dict = vars(self.parse_args())
        
        import json
        with open(os.path.join(path, 'config.json'), 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def print_config(self):
        """Print current configuration"""
        config = self.parse_args()
        print("\n🔧 Training Configuration:")
        print("=" * 30)
        for key, value in sorted(vars(config).items()):
            print(f"{key:20s}: {value}")
        print("=" * 30)

# Convenience function for quick config creation
def get_config():
    """Get a configured TrainConfig instance"""
    return TrainConfig()

# Default configuration values
DEFAULT_CONFIG = {
    'dataset': 'reddit',
    'model': 'sage', 
    'hidden_dim': 256,
    'hidden_layers': 3,
    'maxk': 32,
    'nonlinear': 'maxk',
    'dropout': 0.5,
    'epochs': 1000,
    'w_lr': 0.01,
    'w_weight_decay': 0.0,
    'gpu': 0,
    'seed': 97,
    'norm': False,
    'selfloop': False,
    'enable_lookahead': False,
    'data_path': './data/',
    'log_every': 100,
    'eval_every': 100, 
    'save_every': 500,
    'timing': False,
    'profile': False,
    'debug': False
}

if __name__ == "__main__":
    # Test the configuration
    config = TrainConfig()
    args = config.parse_args()
    config.print_config()
    print(f"\n✅ Config test successful!")
    print(f"Output path: {args.path}")
    print(f"Plot path: {args.plot_path}")