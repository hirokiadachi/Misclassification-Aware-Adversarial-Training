import os
import yaml
import shutil
import argparse
import numpy as np
import multiprocessing

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from advertorch.attacks import PGDAttack

from utils import *
from models.wideresnet import *

def training(epoch, train_dataloader, model, xent, kl, optimizer, lambda_kl=6, num_classes=10, n_steps=10, epsilon=8/255, alpha=2/255):
    model.train()
    print('\nEpoch: %d' % epoch)
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    for idx, (inputs, targets) in enumerate(train_dataloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        onehot = torch.eye(num_classes)[targets].cuda()
        noise = torch.FloatTensor(inputs.shape).uniform_(-epsilon, epsilon).cuda()
        x = torch.clamp(inputs + noise, min=0, max=1)
        
        for _ in range(n_steps):
            x.requires_grad_()
            logits = model(x)
            loss = xent(logits, targets)
            loss.backward()
            grads = x.grad.data
            
            x = x.detach() + alpha * torch.sign(grads.detach())
            x = torch.min(torch.max(x, inputs - epsilon), inputs + epsilon)
            x = torch.clamp(x, min=0, max=1)
        
        logits_nat = model(inputs)
        logits_adv = model(x)
        probs_nat = torch.softmax(logits_nat, dim=1)
        probs_adv = torch.softmax(logits_adv, dim=1)
        tmp = torch.argsort(probs_adv, dim=1)[:, -2:]
        new_y = torch.where(tmp[:, -1] == targets, tmp[:, -2], tmp[:, -1])
        
        bce = xent(logits_adv, targets) + F.nll_loss(torch.log(1 - probs_adv + 1e-12), new_y)
        
        true_probs = torch.gather(probs_nat, dim=1, index=(targets.unsqueeze(1)).long()).squeeze()
        reg_loss = torch.sum(torch.sum(
                kl(torch.log(probs_adv + 1e-12), probs_nat), dim=1) * (1 - true_probs)) / inputs.size(0)
        
        loss = bce + lambda_kl * reg_loss 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        prec1, prec5 = accuracy(logits_adv, targets, topk=(1,5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1, inputs.size(0))
        top5.update(prec5, inputs.size(0))
        
        if idx % 100 == 0:
            print('  Loss: {loss.val: .4f} ({loss.avg: .4f})'
                  '  Acc (top1): {top1.val: .4f} ({top1.avg: .4f})'
                  '  Acc (top5): {top5.val: .4f} ({top5.avg: .4f})'.format(loss=losses, top1=top1, top5=top5))
    return losses.avg, top1.avg

def validation(epoch, test_dataloader, model, xent, n_repeat=10, epsilon=8/255, alpha=2/255):
    model.eval()
    losses_nat = AverageMeter()
    top1_nat = AverageMeter()
    top5_nat = AverageMeter()
    
    losses_adv = AverageMeter()
    top1_adv = AverageMeter()
    top5_adv = AverageMeter()
    
    
    attack = PGDAttack(predict=model, loss_fn=xent, eps=epsilon, nb_iter=n_repeat, 
                       eps_iter=alpha, rand_init=True, clip_min=0.0, clip_max=1.0)
    for idx, (inputs, targets) in enumerate(test_dataloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        b = inputs.size(0)
        onehot = torch.eye(num_classes)[targets].cuda()
        org_img = inputs.clone()
        
        ## Inference natural images
        with torch.no_grad():
            logits_nat = model(inputs)
            loss_nat = xent(logits_nat, targets)
        prec1_nat, prec5_nat = accuracy(logits_nat, targets, topk=(1,5))
        losses_nat.update(loss_nat.item(), inputs.size(0))
        top1_nat.update(prec1_nat, inputs.size(0))
        top5_nat.update(prec5_nat, inputs.size(0))
        
        ## Evaluation adversarial attack
        img_adv = attack(inputs, targets)
        with torch.no_grad():
            logits_adv = model(img_adv)
            loss_adv = xent(logits_adv, targets)
        prec1_adv, prec5_adv = accuracy(logits_adv, targets, topk=(1,5))
                
        losses_adv.update(loss_adv.item(), inputs.size(0))
        top1_adv.update(prec1_adv, inputs.size(0))
        top5_adv.update(prec5_adv, inputs.size(0))
       
    print('natural validation accuracy'
          '  Top1 Acc (AT): {top1.val: .4f} ({top1.avg: .4f})'
          '  Top5 Acc (AT): {top5.val: .4f} ({top5.avg: .4f})'.format(top1=top1_nat, top5=top5_nat))
    print('adversarial validation accuracy'
          '  Top1 Acc (AT): {top1.val: .4f} ({top1.avg: .4f})'
          '  Top5 Acc (AT): {top5.val: .4f} ({top5.avg: .4f})'.format(top1=top1_adv, top5=top5_adv))
    return losses_adv.avg, top1_adv.avg, losses_nat.avg, top1_nat.avg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfile', type=str)
    parser.add_argument('-g', '--gpu', type=str, default='0')
    parser.add_argument('--checkpoint', type=str, default='checkpoint')
    parser.add_argument('--seed_pytorch', type=int, default=np.random.randint(4294967295))
    parser.add_argument('--seed_numpy', type=int, default=np.random.randint(4294967295))
    args = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    np.random.seed(args.seed_numpy)
    torch.manual_seed(args.seed_pytorch)
    
    #####################################################
    # Loading training configures
    #####################################################
    with open(args.cfile) as yml_file:
        config = yaml.safe_load(yml_file.read())['training']
    
    batch_size = config['batch_size']
    lr = config['lr']
    momentum = config['momentum']
    weight_decay = config['weight_decay']
    epochs = config['epochs']
    epsilon_budget = config['epsilon']
    epsilon_step = config['alpha']
    steps = config['m']
    pgd_repeat = config['pgd_repeat']
    num_classes = config['n_cls']
    depth = config['depth']
    widen_factor = config['widen_factor']
    dataset = config['dataset']
    image_size = config['image_size']
    gamma = config['gamma']
    #####################################################
    
    os.makedirs(args.checkpoint, exist_ok=True)
    tb_path = os.path.join(args.checkpoint, 'logs')
    if os.path.exists(tb_path):
        shutil.rmtree(tb_path)
    tb = SummaryWriter(log_dir=tb_path)
    
    best_acc_clean = 0
    best_acc_adv = 0
    
    train_transforms = transforms.Compose([
        transforms.RandomCrop(image_size, padding=4),
        transforms.transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    train_dataset = datasets.CIFAR10('/root/mnt/datasets/data', train=True, download=False, transform=train_transforms)
    test_dataset = datasets.CIFAR10('/root/mnt/datasets/data', train=False, download=False, transform=transforms.ToTensor())
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=multiprocessing.cpu_count())
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=multiprocessing.cpu_count())
    
    model = nn.DataParallel(WideResNet(depth, num_classes, widen_factor, 0.3).cuda())
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = [int(epochs*0.5), int(epochs*0.75)]
    adjust_learning_rate = lr_scheduler.MultiStepLR(optimizer, scheduler, gamma=0.1)
    xent = nn.CrossEntropyLoss()
    kl = nn.KLDivLoss(reduction='none')
    
    for epoch in range(epochs):
        train_loss, train_acc = training(epoch, train_dataloader, model, xent, kl, optimizer)
        test_loss_adv, test_acc_adv, test_loss_clean, test_acc_clean = validation(epoch, test_dataloader, model, xent, pgd_repeat)
        
        tb.add_scalar('train_loss', train_loss, epoch)
        tb.add_scalar('train_acc', train_acc, epoch)
        tb.add_scalars('Loss', {'val/clean': test_loss_clean, 'val/adv': test_loss_adv}, epoch)
        tb.add_scalars('Accuracy', {'val/clean': test_acc_clean, 'val/adv': test_acc_adv}, epoch)
        
        is_best_clean = test_acc_clean > best_acc_clean
        is_best_adv = test_acc_adv > best_acc_adv
        if is_best_clean and is_best_adv:
            best_acc_clean = max(test_acc_clean, best_acc_clean)
            best_acc_adv = max(test_acc_adv, best_acc_adv)
        
        save_checkpoint({
            'epoch': epoch + 1,
            'seed_numpy': args.seed_numpy,
            'seed_pytorch': args.seed_pytorch,
            'state_dict': model.state_dict(),
            'acc_clean': test_acc_clean,
            'acc_adv': test_acc_adv,
            'best_acc_clean': best_acc_clean,
            'best_acc_adv': best_acc_adv,
            'optimizer': optimizer.state_dict()}, is_best_clean, is_best_adv, checkpoint=args.checkpoint)

        adjust_learning_rate.step()