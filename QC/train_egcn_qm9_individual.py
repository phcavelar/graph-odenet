#!/usr/bin/python
# -*- coding: utf-8 -*-

# Adapted from https://github.com/priba/nmp_qc

"""
    Trains different GNN models on various datasets. Methodology defined in:

    Gilmer, J., Schoenholz S.S., Riley, P.F., Vinyals, O., Dahl, G.E. (2017)
    Neural Message Passing for Quantum Chemistry.
    arXiv preprint arXiv:1704.01212 [cs.LG]

"""

import argparse
import os

# Torch
import torch
import torch.optim as optim

# Our Modules
import layer_models as models
from LogMetric import Logger
from util import restricted_float, count_params, train, validate, read_dataset, get_metric_by_task_type, save_checkpoint

__author__ = "Pedro H.C. Avelar, Pau Riba, Anjan Dutta"
__email__ = "phcavelar@inf.ufrgs.br, priba@cvc.uab.cat, adutta@cvc.uab.cat"

# Argument parser
parser = argparse.ArgumentParser(description='Neural message passing')

parser.add_argument('--dataset', choices=["qm9"], default='qm9', help='dataset name, can be any of "qm9", "mutag", "enzymes" or a custom one')
parser.add_argument('--dataset-type', choices=["regression"], help='dataset type')
parser.add_argument('--dataset-path', help='custom dataset path')
parser.add_argument('--log_path', default='./log/{model}-{layers}/{dataset}_individual/{feature}', help='log path')
parser.add_argument('--plotLr', default=False, help='allow plotting the data')
parser.add_argument('--plot_path', default='./plot/{model}-{layers}/{dataset}_individual/{feature}', help='plot path')
parser.add_argument('--resume', default='./checkpoint/{model}-{layers}/{dataset}_individual/{feature}',
                    help='path to latest checkpoint')
# Optimization Options
parser.add_argument('--model', choices=["egcnsum", "egcns2s", "ennsum", "enns2s", "eressum", "eress2s", "eodesum", "eodes2s"], default="egcnsum",
                    help='Which model to train')
parser.add_argument('--hidden', type=int, default=73, metavar='H',
                    help='Number of hidden units in hidden layers(default: 73)')
parser.add_argument('--batch-size', type=int, default=20, metavar='B',
                    help='Input batch size for training (default: 20)')
parser.add_argument('--layers', type=int, default=3, metavar='L',
                    help='Number of layers/message-passing iterations (default: 3)')
parser.add_argument('--s2s', type=int, default=4, metavar='S',
                    help='Number of Set2Set iterations (default: 4)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
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
}

dataset_types = {
        "qm9": "regression",
}

dataset_targets = {
    "qm9": ["mu", "alpha", "homo", "lumo", "gap", "r2", "zpve", "U0", "U", "H", "G", "Cv"]
}

model_dict = {
        "egcnsum": models.EdgeGCN_K_Sum,
        "egcns2s": models.EdgeGCN_K_Set2Set,
        "ennsum": models.MPNN_ENN_K_Sum,
        "enns2s": models.MPNN_ENN_K_Set2Set,
        "eressum": models.UnimplementedModel,
        "eress2s": models.EdgeRES1_K_Set2Set,
        "eodesum": models.UnimplementedModel,
        "eodes2s": models.UnimplementedModel,
}


def main():
    global args, best_er1
    args = parser.parse_args()

    # Check if CUDA is enabled
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    
    for tgt_idx, tgt in enumerate( dataset_targets[args.dataset] ):
        print("Training a model for {}".format(tgt))
    
        # Load data
        root = args.dataset_path if args.dataset_path else dataset_paths[args.dataset]
        task_type = args.dataset_type if args.dataset_type else dataset_types[args.dataset]
        if args.resume:
            resume_dir = args.resume.format(dataset=args.dataset,model=args.model,layers=args.layers,feature=tgt)
        #end if
        Model_Class = model_dict[args.model]

        print("Preparing dataset")
        node_features, edge_features, target_features, task_type, train_loader, valid_loader, test_loader = read_dataset(args.dataset,root,args.batch_size,args.prefetch)

        # Define model and optimizer

        print('\tCreate model')
        hidden_state_size = args.hidden
        model = Model_Class(node_features=node_features, edge_features=edge_features, target_features=1, hidden_features=hidden_state_size, num_layers=args.layers, dropout=0.5, type=task_type, s2s_processing_steps=args.s2s)
        print("#Parameters: {param_count}".format(param_count=count_params(model)))

        print('Optimizer')
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
        criterion, evaluation, metric_name, metric_compare, metric_best = get_metric_by_task_type(task_type,target_features)

        print('Logger')
        logger = Logger(args.log_path.format(dataset=args.dataset,model=args.model,layers=args.layers,feature=tgt))

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
            try:
                if epoch > args.epochs * args.schedule[0] and epoch < args.epochs * args.schedule[1]:
                    args.lr -= lr_step
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = args.lr
                #end if

                # train for one epoch
                train(train_loader, model, criterion, optimizer, epoch, evaluation, logger, target_range=(tgt_idx,), tgt_name=tgt, metric_name=metric_name, cuda=args.cuda, log_interval=args.log_interval)

                # evaluate on test set
                er1 = validate(valid_loader, model, criterion, evaluation, logger, target_range=(tgt_idx,), tgt_name=tgt, metric_name=metric_name, cuda=args.cuda, log_interval=args.log_interval)

                is_best = metric_compare(er1, best_er1)
                best_er1 = metric_best(er1, best_er1)
                save_checkpoint({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'best_er1': best_er1,
                                       'optimizer': optimizer.state_dict(), }, is_best=is_best, directory=resume_dir)

                # Logger step
                logger.log_value('learning_rate', args.lr).step()
            except KeyboardInterrupt:
                break
            #end try
        #end for

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
            #end if
        #end if

        # (For testing)
        validate(test_loader, model, criterion, evaluation, target_range=(tgt_idx,), tgt_name=tgt, metric_name=metric_name, cuda=args.cuda, log_interval=args.log_interval)
    #end for tgt
#end main

if __name__ == '__main__':
    main()
