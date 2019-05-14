# Adapted from https://github.com/priba/nmp_qc

import os
import time
import argparse

import numpy as np
import shutil

import torch
import torch.nn as nn
from torch.autograd import Variable

# Our Modules
import datasets
from LogMetric import AverageMeter
from GraphReader.graph_reader import create_graph_mutag, divide_datasets

def save_checkpoint(state, is_best, directory):

    if not os.path.isdir(directory):
        os.makedirs(directory)
    checkpoint_file = os.path.join(directory, 'checkpoint.pth')
    best_model_file = os.path.join(directory, 'model_best.pth')
    torch.save(state, checkpoint_file)
    if is_best:
        shutil.copyfile(checkpoint_file, best_model_file)

def get_metric_by_task_type(task_type,target_features):
    if task_type == "regression":
        criterion = nn.MSELoss(reduction='mean')
        evaluation = lambda output, target: torch.mean(torch.abs(output - target))
        metric_name = "Error ratio"
        metric_compare = lambda x, y: x<y
        metric_best = lambda x, y: min(x,y)
    elif task_type == "classification":
        if target_features == 1:
            criterion = nn.BCEWithLogitsLoss(reduction='mean')
            evaluation = lambda output, target: torch.mean(torch.sigmoid(output).round().eq(target).double())
            metric_name = "Accuracy"
            metric_compare = lambda x, y: x>y
            metric_best = lambda x, y: max(x,y)
        else:
            criterion = nn.NLLLoss(reduction="mean")
            evaluation = lambda output, target: torch.mean(output.max(1).type_as(target).eq(target).double())
            metric_name = "Accuracy"
            metric_compare = lambda x, y: x>y
            metric_best = lambda x, y: max(x,y)
        #end if
    else:
        raise ValueError( "Unrecognised task type" )
    #end if
    return criterion, evaluation, metric_name, metric_compare, metric_best
#end get_metric_by_task_type

def restricted_float(x, inter):
    x = float(x)
    if x < inter[0] or x > inter[1]:
        raise argparse.ArgumentTypeError("%r not in range [1e-5, 1e-4]"%(x,))
    return x

def read_dataset(dataset,root,batch_size,num_workers):
    if dataset=="qm9":
        print('Prepare files')
        
        files = [f for f in os.listdir(root) if os.path.isfile(os.path.join(root, f))]

        idx = np.random.permutation(len(files))
        idx = idx.tolist()

        valid_ids = [files[i] for i in idx[0:10000]]
        test_ids = [files[i] for i in idx[10000:20000]]
        train_ids = [files[i] for i in idx[20000:]]

        data_train = datasets.Qm9(root, train_ids, edge_transform=datasets.utils.qm9_edges, e_representation='raw_distance')
        data_valid = datasets.Qm9(root, valid_ids, edge_transform=datasets.utils.qm9_edges, e_representation='raw_distance')
        data_test = datasets.Qm9(root, test_ids, edge_transform=datasets.utils.qm9_edges, e_representation='raw_distance')

        # Select one graph
        g_tuple, l = data_train[0]
        g, h_t, e = g_tuple
        node_features = len(h_t[0])
        edge_features = len(list(e.values())[0])
        target_features = len(l)
        task_type ='regression'

        print('\tStatistics')
        #stat_dict = datasets.utils.get_graph_stats(data_valid, ['target_mean', 'target_std'])
        stat_dict = {}
        stat_dict['target_mean'] = np.array([2.71802732e+00,   7.51685080e+01,  -2.40259300e-01,   1.09503300e-02,
                                             2.51209430e-01,   1.18997445e+03,   1.48493130e-01,  -4.11609491e+02,
                                            -4.11601022e+02,  -4.11600078e+02,  -4.11642909e+02,   3.15894998e+01])
        stat_dict['target_std'] = np.array([1.58422291e+00,   8.29443552e+00,   2.23854977e-02,   4.71030547e-02,
                                            4.77156393e-02,   2.80754665e+02,   3.37238236e-02,   3.97717205e+01,
                                            3.97715029e+01,   3.97715029e+01,   3.97722334e+01,   4.09458852e+00])

        data_train.set_target_transform(lambda x: datasets.utils.normalize_data(x,stat_dict['target_mean'],
                                                                                stat_dict['target_std']))
        data_valid.set_target_transform(lambda x: datasets.utils.normalize_data(x, stat_dict['target_mean'],
                                                                                stat_dict['target_std']))
        data_test.set_target_transform(lambda x: datasets.utils.normalize_data(x, stat_dict['target_mean'],
                                                                               stat_dict['target_std']))
    elif dataset=="mutag":
        files = [f for f in os.listdir(root) if os.path.isfile(os.path.join(root, f))]
        classes = [create_graph_mutag(os.path.join(root, f))[1] for f in files]
            
        train_ids, train_classes, valid_ids, valid_classes, test_ids, test_classes = divide_datasets(files, classes)

        data_train = datasets.MUTAG(root, train_ids, train_classes)
        data_valid = datasets.MUTAG(root, valid_ids, valid_classes)
        data_test = datasets.MUTAG(root, test_ids, test_classes)
        
        # Select one graph
        g_tuple, l = data_train[0]
        g, h_t, e = g_tuple
        node_features = len(h_t[0])
        edge_features = len(list(e.values())[0])
        target_features = len(l)
        task_type ='classification'
    elif dataset=="enzymes":
        raise NotImplementedError("Enzymes not yet implemented")
    else:
        raise NotImplementedError("General loading not yet implemented")
    #end

    # Data Loader
    train_loader = torch.utils.data.DataLoader(data_train,
                                               batch_size=batch_size, shuffle=True,
                                               collate_fn=datasets.utils.collate_g_concat_edge_data,
                                               num_workers=num_workers, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(data_valid,
                                               batch_size=batch_size, shuffle=False,
                                               collate_fn=datasets.utils.collate_g_concat_edge_data,
                                               num_workers=num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(data_test,
                                              batch_size=batch_size, shuffle=False,
                                              collate_fn=datasets.utils.collate_g_concat_edge_data,
                                              num_workers=num_workers, pin_memory=True)
    return node_features, edge_features, target_features, task_type, train_loader, valid_loader, test_loader
#end read_dataset

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
#end count_params

def train(train_loader, model, criterion, optimizer, epoch, evaluation, logger, target_range=(0,None), tgt_name="", metric_name="metric", cuda=False, log_interval=20):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    metric = AverageMeter()
    # switch to train mode
    model.train()

    end = time.time()
    for i, (batch_size,g,b,x,e_d,e_src,e_tgt,target) in enumerate(train_loader):
        #e_tgt = e_tgt.to_sparse()
        if len(target_range) == 1:
            target=target[:,target_range]
        elif len(target_range) == 2:
          target = target[:,target_range[0]:target_range[1]]
        elif len(target_range) == 3:
          target = target[:,target_range[0]:target_range[1]:target_range[2]]
        #end if
        
        if cuda:
            g,b,x,e_d,e_src,e_tgt,target = map(lambda a:a.cuda(), (g,b,x,e_d,e_src,e_tgt,target))
        #g,b,x,e_d,e_src,e_tgt,target = map(lambda a: torch.tensor(a,requires_grad=True if a.dtype == torch.float else False,device=a.device), (g,b,x,e_d,e_src,e_tgt,target))
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
        metric.update(evaluation(output, target).item(), batch_size)

        # compute gradient and do SGD step
        train_loss.backward()
        optimizer.step()
  
        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % log_interval == 0 and i > 0:

            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  '{metric_name} {metric.val:.4f} ({metric.avg:.4f})'
                  .format(epoch, i, len(train_loader), batch_time=batch_time,
                          data_time=data_time, loss=losses, metric_name=metric_name, metric=metric), flush=True)
                          
    logger.log_value('train_epoch_loss', losses.avg)
    logger.log_value('train_epoch_{metric}'.format(metric=metric_name), metric.avg)

    print('Epoch: [{0}] {tgt_name} Avg {metric_name} {metric.avg:.3f}; Average Loss {loss.avg:.3f}; Avg Time x Batch {b_time.avg:.3f}'
          .format(epoch, metric_name=metric_name, metric=metric, loss=losses, b_time=batch_time, tgt_name=tgt_name,), flush=True)


def validate(val_loader, model, criterion, evaluation, logger=None, target_range=(0,None), tgt_name="", metric_name="metric", cuda=False, log_interval=20):
    batch_time = AverageMeter()
    losses = AverageMeter()
    metric = AverageMeter()

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (batch_size,g,b,x,e_d,e_src,e_tgt,target) in enumerate(val_loader):
            e_tgt.to_sparse()
            if target_range is not None:
                if len(target_range) == 1:
                    target=target[:,target_range[0]]
                elif len(target_range) == 2:
                  target = target[:,target_range[0]:target_range[1]]
                elif len(target_range) == 3:
                  target = target[:,target_range[0]:target_range[1]:target_range[2]]
                #end if
            #end if
            
            if cuda:
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
            metric.update(evaluation(output, target).item(), batch_size)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % log_interval == 0 and i > 0:
                
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      '{metric_name} {metric.val:.4f} ({metric.avg:.4f})'
                      .format(i, len(val_loader), batch_time=batch_time,
                              loss=losses, metric_name=metric_name, metric=metric), flush=True)
            #end if
        #end for
    #end torch.no_grad

    print(' * {tgt_name} Average {metric_name} {metric.avg:.3f}; Average Loss {loss.avg:.3f}'
          .format(metric_name=metric_name, metric=metric, loss=losses, tgt_name=tgt_name,), flush=True)

    if logger is not None:
        logger.log_value('test_epoch_loss', losses.avg)
        logger.log_value('test_epoch_{metric}'.format(metric=metric_name), metric.avg)

    return metric.avg

