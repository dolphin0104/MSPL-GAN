import shutil
import os
from config import train_config as args
from trainer import Trainer


def main():
    if args.CONFIG_FILE is not None:
        if not os.path.exists(args.Paths['save_dir']): os.makedirs(args.Paths['save_dir'])
        save_configs = os.path.join(args.Paths['save_dir'], 'Training_CONFIG.py')
        shutil.copy(args.CONFIG_FILE, save_configs)
    trainer = Trainer(args)
    trainer.train()

if __name__ == '__main__':
    main()