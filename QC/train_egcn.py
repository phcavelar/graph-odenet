#!/usr/bin/python
# -*- coding: utf-8 -*-

# Adapted from https://github.com/priba/nmp_qc

"""
    Trains different GNN models on various datasets. Methodology defined in:

    Gilmer, J., Schoenholz S.S., Riley, P.F., Vinyals, O., Dahl, G.E. (2017)
    Neural Message Passing for Quantum Chemistry.
    arXiv preprint arXiv:1704.01212 [cs.LG]

"""

# Torch
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable

import time
import argparse
import os
import numpy as np

# Our Modules
import datasets
from datasets import utils
import models
from LogMetric import AverageMeter, Logger

__author__ = "Pedro H.C. Avelar, Pau Riba, Anjan Dutta"
__email__ = "phcavelar@inf.ufrgs.br, priba@cvc.uab.cat, adutta@cvc.uab.cat"


# Parser check
def restricted_float(x, inter):
    x = float(x)
    if x < inter[0] or x > inter[1]:
        raise argparse.ArgumentTypeError("%r not in range [1e-5, 1e-4]"%(x,))
    return x

# Argument parser
parser = argparse.ArgumentParser(description='Neural message passing')

parser.add_argument('--dataset', default='qm9', help='dataset name, can be any of "qm9", "mutag", "enzymes" or a custom one')
parser.add_argument('--dataset-type', choices=["classification", "regression"], help='dataset name')
parser.add_argument('--dataset-path', help='custom dataset path')
parser.add_argument('--log_path', default='./log/{model}/{dataset}/', help='log path')
parser.add_argument('--plotLr', default=False, help='allow plotting the data')
parser.add_argument('--plot_path', default='./plot/{model}/{dataset}/', help='plot path')
parser.add_argument('--resume', default='./checkpoint/{model}/{dataset}/',
                    help='path to latest checkpoint')
# Optimization Options
parser.add_argument('--model', choices=["egcn3sum", "egcn3s2s", "ennsum", "enns2s", "eres3sum", "eres3s2s", "eode3sum", "eode3s2s"], default="egc3",
                    help='Which model to train')
parser.add_argument('--batch-size', type=int, default=20, metavar='N',
                    help='Input batch size for training (default: 20)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Enables CUDA training')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='Number of epochs to train (default: 100)')
parser.add_argument('--lr', type=lambda x: restricted_float(x, [1e-5, 1e-2]), default=1e-3, metavar='LR',
                    help='Initial learning rate [1e-5, 5e-4] (default: 1e-4)')
parser.add_argument('--lr-decay', type=lambda x: restricted_float(x, [.01, 1]), default=0.6, metavar='LR-DECAY',
                    help='Learning rate decay factor [.01, 1] (default: 0.6)')
parser.add_argument('--weight-decay', type=lambda x: restricted_float(x, [0, 1]), default=5e-4, metavar='WEIGHT-DECAY',
                    help='Learning rate decay factor [0, 1] (default: 5e-4)')
parser.add_argument('--schedule', type=list, default=[0.2, 0.8], metavar='S',
                    help='Percentage of epochs to start the learning rate decay [0, 1] (default: [0.1, 0.9])')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
# i/o
parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                    help='How many batches to wait before logging training status')
# Accelerating
parser.add_argument('--prefetch', type=int, default=8, help='Pre-fetching threads.')

best_er1 = 0

dataset_paths = {
        "qm9": "./data/qm9/dsgdb9nsd/",
        "mutag": "./data/mutag/",
        "enzymes": "./data/enzymes/",
}

dataset_types = {
        "qm9": "regression",
        "mutag": "classification",
        "enzymes": "classification",
}

class UnimplementedModel(nn.Module):
    def __init__(self,*args,**kwargs):
        raise NotImplementedError("Model not implemented yet")
    def forward(self,*args,**kwargs):
        raise NotImplementedError("Model not implemented yet")
#end UnimplementedModel

model_dict = {
        "egcn3sum": models.EdgeGCN3_Sum,
        "egcn3s2s": models.EdgeGCN3_Set2Set,
        "ennsum": models.MPNN_ENN_Sum,
        "enns2s": models.MPNN_ENN_Set2Set,
        "eres3sum": UnimplementedModel,
        "eres3s2s": UnimplementedModel,
        "eode3sum": UnimplementedModel,
        "eode3s2s": UnimplementedModel,
}

def main():

    global args, best_er1
    args = parser.parse_args()

    # Check if CUDA is enabled
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # Load data
    root = args.dataset_path if args.dataset_path else dataset_paths[args.dataset]
    task_type = args.dataset_type if args.dataset_type else dataset_types[args.dataset]
    if args.resume:
        resume_dir = args.resume.format(dataset=args.dataset,model=args.model)
    #end if
    Model_Class = model_dict[args.model]

    print('Prepare files')
    files = [f for f in os.listdir(root) if os.path.isfile(os.path.join(root, f))]

    idx = np.random.permutation(len(files))
    idx = idx.tolist()

    valid_ids = [files[i] for i in idx[0:10000]]
    test_ids = [files[i] for i in idx[10000:20000]]
    train_ids = [files[i] for i in idx[20000:]]

    data_train = datasets.Qm9(root, train_ids, edge_transform=utils.qm9_edges, e_representation='raw_distance')
    data_valid = datasets.Qm9(root, valid_ids, edge_transform=utils.qm9_edges, e_representation='raw_distance')
    data_test = datasets.Qm9(root, test_ids, edge_transform=utils.qm9_edges, e_representation='raw_distance')

    # Define model and optimizer
    print('Define model')
    # Select one graph
    g_tuple, l = data_train[0]
    g, h_t, e = g_tuple

    print('\tCreate model')
    in_n = len(h_t[0])
    in_e = len(list(e.values())[0])
    hidden_state_size = 73
    n_layers = 3
    l_target = len(l)
    type ='regression'
    model = Model_Class(in_n, in_e, hidden_state_size, l_target, dropout=0.5, type=type)
    del in_n, hidden_state_size, n_layers, l_target, type

    #print('Check cuda for model')
    #if args.cuda:
    #    print('\t* Cuda')
    #    model = model.cuda()

    print('\tStatistics')
    stat_dict = datasets.utils.get_graph_stats(data_valid, ['target_mean', 'target_std'])

    data_train.set_target_transform(lambda x: datasets.utils.normalize_data(x,stat_dict['target_mean'],
                                                                            stat_dict['target_std']))
    data_valid.set_target_transform(lambda x: datasets.utils.normalize_data(x, stat_dict['target_mean'],
                                                                            stat_dict['target_std']))
    data_test.set_target_transform(lambda x: datasets.utils.normalize_data(x, stat_dict['target_mean'],
                                                                           stat_dict['target_std']))

    # Data Loader
    train_loader = torch.utils.data.DataLoader(data_train,
                                               batch_size=args.batch_size, shuffle=True,
                                               collate_fn=datasets.utils.collate_g_concat_edge_data,
                                               num_workers=args.prefetch, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(data_valid,
                                               batch_size=args.batch_size, shuffle=False,
                                               collate_fn=datasets.utils.collate_g_concat_edge_data,
                                               num_workers=args.prefetch, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(data_test,
                                              batch_size=args.batch_size, shuffle=False,
                                              collate_fn=datasets.utils.collate_g_concat_edge_data,
                                              num_workers=args.prefetch, pin_memory=True)

    print('Optimizer')
    optimizer = optim.Adam(model.parameters(), lr=args.lr)#, weight_decay=args.weight_decay)

    criterion = nn.MSELoss(reduction='mean')

    evaluation = lambda output, target: torch.mean(torch.abs(output - target) / torch.abs(target))

    print('Logger')
    logger = Logger(args.log_path.format(dataset=args.dataset,model=args.model))

    lr_step = (args.lr-args.lr*args.lr_decay)/(args.epochs*args.schedule[1] - args.epochs*args.schedule[0])

    # get the best checkpoint if available without training
    if args.resume:
        checkpoint_dir = resume_dir
        best_model_file = os.path.join(checkpoint_dir, 'model_best.pth')
        if not os.path.isdir(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        if os.path.isfile(best_model_file):
            print("=> loading best model '{}'".format(best_model_file))
            checkpoint = torch.load(best_model_file)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_er1']
            model.load_state_dict(checkpoint['state_dict'])
            if args.cuda:
                model.cuda()
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded best model '{}' (epoch {})".format(best_model_file, checkpoint['epoch']))
        else:
            print("=> no best model found at '{}'".format(best_model_file))

    print('Check cuda')
    if args.cuda:
        print('\t* Cuda')
        model = model.cuda()
        criterion = criterion.cuda()

    # Epoch for loop
    for epoch in range(0, args.epochs):

        if epoch > args.epochs * args.schedule[0] and epoch < args.epochs * args.schedule[1]:
            args.lr -= lr_step
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, evaluation, logger)

        # evaluate on test set
        er1 = validate(valid_loader, model, criterion, evaluation, logger)

        is_best = er1 > best_er1
        best_er1 = min(er1, best_er1)
        utils.save_checkpoint({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'best_er1': best_er1,
                               'optimizer': optimizer.state_dict(), }, is_best=is_best, directory=resume_dir)

        # Logger step
        logger.log_value('learning_rate', args.lr).step()

    # get the best checkpoint and test it with test set
    if args.resume:
        checkpoint_dir = resume_dir
        best_model_file = os.path.join(checkpoint_dir, 'model_best.pth')
        if not os.path.isdir(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        if os.path.isfile(best_model_file):
            print("=> loading best model '{}'".format(best_model_file))
            checkpoint = torch.load(best_model_file)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_er1']
            model.load_state_dict(checkpoint['state_dict'])
            if args.cuda:
                model.cuda()
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded best model '{}' (epoch {})".format(best_model_file, checkpoint['epoch']))
        else:
            print("=> no best model found at '{}'".format(best_model_file))

    # For testing
    validate(test_loader, model, criterion, evaluation)


def train(train_loader, model, criterion, optimizer, epoch, evaluation, logger):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    error_ratio = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (batch_size,g,b,x,e_d,e_src,e_tgt,target) in enumerate(train_loader):
        e_tgt = e_tgt.to_sparse()
        
        if args.cuda:
            g,b,x,e_d,e_src,e_tgt,target = map(lambda a:a.cuda(), (g,b,x,e_d,e_src,e_tgt,target))
        #g,b,x,e_d,e_src,e_tgt,target = map(lambda a:Variable(a), (g,b,x,e_d,e_src,e_tgt,target))

        # Measure data loading time
        data_time.update(time.time() - end)

        optimizer.zero_grad()
        # Compute output
        train_loss = torch.zeros((),)
        output = model(
                node_features=x,
                edge_features=e_d,
                Esrc=e_src,
                Etgt=e_tgt,
                batch=b
                )
        train_loss = criterion(output, target)

        # Logs
        losses.update(train_loss.item(), batch_size)
        error_ratio.update(evaluation(output, target).item(), batch_size)

        # compute gradient and do SGD step
        train_loss.backward()
        optimizer.step()

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.log_interval == 0 and i > 0:

            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Error Ratio {err.val:.4f} ({err.avg:.4f})'
                  .format(epoch, i, len(train_loader), batch_time=batch_time,
                          data_time=data_time, loss=losses, err=error_ratio), flush=True)
                          
    logger.log_value('train_epoch_loss', losses.avg)
    logger.log_value('train_epoch_error_ratio', error_ratio.avg)

    print('Epoch: [{0}] Avg Error Ratio {err.avg:.3f}; Average Loss {loss.avg:.3f}; Avg Time x Batch {b_time.avg:.3f}'
          .format(epoch, err=error_ratio, loss=losses, b_time=batch_time), flush=True)


def validate(val_loader, model, criterion, evaluation, logger=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    error_ratio = AverageMeter()

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (batch_size,g,b,x,e_d,e_src,e_tgt,target) in enumerate(val_loader):
            e_tgt.to_sparse()
            
            if args.cuda:
                g,b,x,e_d,e_src,e_tgt,target = map(lambda a:a.cuda(), (g,b,x,e_d,e_src,e_tgt,target))
            #g,b,x,e_d,e_src,e_tgt,target = map(lambda a:Variable(a), (g,b,x,e_d,e_src,e_tgt,target))
            
            # Compute output
            train_loss = torch.zeros((),)
            output = model(
                    node_features=x,
                    edge_features=e_d,
                    Esrc=e_src,
                    Etgt=e_tgt,
                    batch=b
                    )

            # Logs
            losses.update(criterion(output, target).item(), batch_size)
            error_ratio.update(evaluation(output, target).item(), batch_size)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.log_interval == 0 and i > 0:
                
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Error Ratio {err.val:.4f} ({err.avg:.4f})'
                      .format(i, len(val_loader), batch_time=batch_time,
                              loss=losses, err=error_ratio), flush=True)
            #end if
        #end for
    #end torch.no_grad

    print(' * Average Error Ratio {err.avg:.3f}; Average Loss {loss.avg:.3f}'
          .format(err=error_ratio, loss=losses), flush=True)

    if logger is not None:
        logger.log_value('test_epoch_loss', losses.avg)
        logger.log_value('test_epoch_error_ratio', error_ratio.avg)

    return error_ratio.avg

    
if __name__ == '__main__':
    main()
