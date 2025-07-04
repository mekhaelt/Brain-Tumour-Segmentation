# train.py
#!/usr/bin/env	python3

""" train network using pytorch
    Cecilia Diana-Albelda
"""

import os
import time

import numpy as np
import torch
import torch.optim as optim
from torch.utils import tensorboard
#from dataset import *
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import cfg
import function
from conf import settings
#from models.discriminatorlayer import discriminator
from dataset import *
from utils import *
import warnings

warnings.filterwarnings(
    "ignore",
    message="TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class.*",
    category=UserWarning
)

def main():

    args = cfg.parse_args()
    # args.four_chan = True  # Override default since we're using 4-channel BRATS data
    # args.thd = True  # Enable 3D processing for medical imaging data
    # args.data_path = 'data'

    GPUdevice = torch.device('cuda', args.gpu_device)
    net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution = args.distributed)
    if args.pretrain:
        weights = torch.load(args.pretrain)
        net.load_state_dict(weights,strict=False)

    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=800, gamma=0.1) #learning rate decay

    '''load pretrained model'''
    if args.weights != 0:
        print(f'=> resuming from {args.weights}')
        assert os.path.exists(args.weights)
        checkpoint_file = os.path.join(args.weights)
        assert os.path.exists(checkpoint_file)
        loc = 'cuda:{}'.format(args.gpu_device)
        checkpoint = torch.load(checkpoint_file, map_location=loc)
        start_epoch = checkpoint['epoch']
        best_tol = checkpoint['best_tol']
        net.load_state_dict(checkpoint['state_dict'],strict=False)
        optimizer.load_state_dict(checkpoint['optimizer'])

        args.path_helper = checkpoint['path_helper']
        logger = create_logger(args.path_helper['log_path'])
        print(f'=> loaded checkpoint {checkpoint_file} (epoch {start_epoch})')

    args.path_helper = set_log_dir('logs', args.exp_name) # set_log_dir from utils.py
    logger = create_logger(args.path_helper['log_path'])
    logger.info(args)


    '''segmentation data'''

    train_transforms = Compose(
            [
                CropForegroundd(keys=["image", "label"], source_key="image"),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                
            ]
        )

    if 'brats' in args.dataset: # BraTS dataset 
        '''Brain Tumor data'''
        brats_train_dataset = Brats(args, args.data_path, mode = 'Training' , transform = train_transforms)
        brats_test_dataset = Brats(args, args.data_path, mode = 'Validation' , transform =  train_transforms)

        dataset_size = len(brats_train_dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(0.2 * dataset_size))
        np.random.shuffle(indices)
        train_sampler = SubsetRandomSampler(indices[split:])
        test_sampler = SubsetRandomSampler(indices[:split])

        nice_train_loader = DataLoader(brats_train_dataset, batch_size=args.b, sampler = train_sampler,  num_workers=6, pin_memory=True)
        nice_test_loader = DataLoader(brats_test_dataset, batch_size=args.b, sampler=test_sampler,  num_workers=6, pin_memory=True)
        
        '''end'''


    '''checkpoint path and tensorboard'''
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)

    # Use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)

    writer = tensorboard.writer.SummaryWriter(log_dir=os.path.join(
        settings.LOG_DIR, args.net, args.path_helper['log_path'].split('\\')[1]))


    '''begain training'''
    best_acc = 0.0
    best_tol = 1e4
    best_dice = 0.0
    best_loss = 10000.0

    for epoch in range(settings.EPOCH):
        net.train()
        time_start = time.time()
        loss, current_lr = function.train_sam(args, net, optimizer, nice_train_loader, epoch, writer, vis = args.vis)
        logger.info(f'Train loss: {loss} || @ epoch {epoch} || @ lr {current_lr}.')
        writer.add_scalar("Train_Loss", loss, epoch)

        # if loss < best_loss:
        #     print('SAVING CHECKPOINT! - BEST LOSS')
        #     best_loss = loss
        #     is_best = True

        #     save_checkpoint({
        #     'epoch': epoch + 1,
        #     'model': args.net,
        #     'state_dict': net.state_dict(),
        #     'optimizer': optimizer.state_dict(),
        # # 'best_tol': tol,
        #     'path_helper': args.path_helper,
        # }, is_best, args.path_helper['ckpt_path'], filename="best_loss")
        # else:
        #     is_best = False

        
        net.eval()
        if epoch: 
            tol, (eiou, edice) = function.validation_sam(args, nice_test_loader, epoch, net, writer)
            logger.info(f'Total score: {tol}, IOU: {eiou}, DICE: {edice} || @ epoch {epoch}.')
            writer.add_scalar("Dice_Valid", edice, epoch)

            if args.distributed != 'none':
                sd = net.module.state_dict()
            else:
                sd = net.state_dict()

            if edice > best_dice:
                print('SAVING CHECKPOINT! - BEST DICE')
                print('CURR: BEST DICE', best_dice)
                best_dice = edice
                is_best = True

                if not os.path.exists(checkpoint_path):
                    os.makedirs(checkpoint_path)

                save_checkpoint({
                'epoch': epoch + 1,
                'model': args.net,
                'state_dict': sd,
                'optimizer': optimizer.state_dict(),
                'best_tol': tol,
                'path_helper': args.path_helper,
            }, is_best, args.path_helper['ckpt_path'], filename="best_dice")
                
                save_checkpoint({
                'epoch': epoch + 1,
                'model': args.net,
                'state_dict': sd,
                'optimizer': optimizer.state_dict(),
                'best_tol': tol,
                'path_helper': args.path_helper,
            }, is_best, checkpoint_path, filename="best_dice.pth")
                
            else:
                is_best = False

    logger.info(f"Best mean Dice score achieved: {best_dice:.4f}")

    writer.close()

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()  # Optional, but safe
    main()
