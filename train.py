import sys
import os
from optparse import OptionParser
import numpy as np
import random
import time
import os
import cv2

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim

from eval import eval_net
from unet import UNet
from utils import get_ids, split_ids, split_train_val, get_imgs_and_masks, batch, read_image, read_masks


def train_net(net,
              epochs=5,
              batch_size=1,
              lr=0.01,
              val_percent=0.05,
              save_cp=True,
              gpu=True):

        # Define directories
    dir_img = 'E:/Dataset/Dataset10k/images/training/'
    dir_mask = 'E:/Dataset/Dataset10k/annotations/training/'

    val_dir_img = 'E:/Dataset/Dataset10k/images/validation/'
    val_dir_mask = 'E:/Dataset/Dataset10k/annotations/validation/'

    dir_checkpoint = 'checkpoints/'

    # Get list of images and annotations
    train_images = os.listdir(dir_img)
    train_masks = os.listdir(dir_mask)
    train_size = len(train_images)

    val_images = os.listdir(val_dir_img)
    val_masks = os.listdir(val_dir_mask)
    val_size = len(val_images)

    val_imgs = np.array([read_image(val_dir_img + i)
                         for i in val_images]).astype(np.float32)
    val_true_masks = np.array([read_masks(val_dir_mask + i)
                               for i in val_masks])
    val = zip(val_imgs, val_true_masks)

    print('''
    Starting training:
        Epochs: {}
        Batch size: {}
        Learning rate: {}
        Training size: {}
        Validation size: {}
        Checkpoints: {}
        CUDA: {}
    '''.format(epochs, batch_size, lr, train_size,
               val_size, str(save_cp), str(gpu)))

    # Define optimizer and loss functions
    optimizer = optim.SGD(net.parameters(),
                          lr=lr,
                          momentum=0.9,
                          weight_decay=0.0005)

    criterion = nn.BCELoss()

    # Start training epochs
    for epoch in range(epochs):
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
        net.train()

        epoch_loss = 0

        for i in range(round(train_size // batch_size)):
            imgs = train_images[i:i+batch_size]
            true_masks = train_masks[i:i+batch_size]

            imgs = np.array([read_image(dir_img + i)
                             for i in imgs]).astype(np.float32)
            true_masks = np.array([read_masks(dir_mask + i)
                                   for i in true_masks])

            imgs = torch.from_numpy(imgs)
            true_masks = torch.from_numpy(true_masks)

            print(imgs.size(), true_masks.size())

            if gpu:
                imgs = imgs.cuda()
                true_masks = true_masks.cuda()

            masks_pred = net(imgs)
            print(masks_pred.size())
            
            masks_probs_flat = masks_pred.view(-1)
            print(masks_probs_flat.size())
            
            true_masks_flat = true_masks.view(-1)
            print(true_masks_flat.size())

            loss = criterion(masks_probs_flat, true_masks_flat)
            epoch_loss += loss.item()

            print('{0:.4f} --- loss: {1:.6f}'.format(i * batch_size, loss.item()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Epoch finished ! Loss: {}'.format(np.mean(epoch_loss)))

        if 1:
            val_dice = eval_net(net, val, gpu)
            print('Validation Dice Coeff: {}'.format(val_dice))

        if save_cp:
            torch.save(net.state_dict(),
                       dir_checkpoint + 'CP{}.pth'.format(epoch + 1))
            print('Checkpoint {} saved !'.format(epoch + 1))


def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=30, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=4,
                      type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.01,
                      type='float', help='learning rate')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=True, help='use cuda')
    parser.add_option('-c', '--load', dest='load',
                      default=False, help='load file model')
    parser.add_option('-s', '--scale', dest='scale', type='float',
                      default=0.5, help='downscaling factor of the images')

    (options, args) = parser.parse_args()
    return options


if __name__ == '__main__':

        # Get arguments/parameters
    args = get_args()

    # Define model/U-Net
    net = UNet(n_channels=3, n_classes=3)

    # Load saved model if args.load is True
    if args.load:
        net.load_state_dict(torch.load(args.load))
        print('Model loaded from {}'.format(args.load))

        # Run model on GPU if args.gpu is True
    if args.gpu:
        net.cuda()
        # cudnn.benchmark = True # faster convolutions, but more memory

    try:
        start = time.time()    # Start time of training

        # Train the model
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  gpu=args.gpu)

        # Show total learning time
        print("Learning time:", time.time()-start, "seconds")

    except KeyboardInterrupt:

            # Save model if training interrupted
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')

        # Exit training
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
