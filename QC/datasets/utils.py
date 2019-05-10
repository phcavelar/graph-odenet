#!/usr/bin/python
# -*- coding: utf-8 -*-

# Adapted from https://github.com/priba/nmp_qc

"""
    utils.py: Functions to process dataset graphs.

    Usage:

"""

from __future__ import print_function

import rdkit
import torch
from joblib import Parallel, delayed
import multiprocessing
import networkx as nx
import numpy as np
import shutil
import os

__author__ = "Pedro HC Avelar, Pau Riba, Anjan Dutta"
__email__ = "phcavelar@inf.ufrgs.br, priba@cvc.uab.cat, adutta@cvc.uab.cat"


def qm9_nodes(g, hydrogen=False):
    h = []
    for n, d in g.nodes(data=True):
        h_t = []
        # Atom type (One-hot H, C, N, O F)
        h_t += [int(d['a_type'] == x) for x in ['H', 'C', 'N', 'O', 'F']]
        # Atomic number
        h_t.append(d['a_num'])
        # Partial Charge
        h_t.append(d['pc'])
        # Acceptor
        h_t.append(d['acceptor'])
        # Donor
        h_t.append(d['donor'])
        # Aromatic
        h_t.append(int(d['aromatic']))
        # Hybradization
        h_t += [int(d['hybridization'] == x) for x in [rdkit.Chem.rdchem.HybridizationType.SP, rdkit.Chem.rdchem.HybridizationType.SP2, rdkit.Chem.rdchem.HybridizationType.SP3]]
        # If number hydrogen is used as a
        if hydrogen:
            h_t.append(d['num_h'])
        h.append(h_t)
    return h


def qm9_edges(g, e_representation='raw_distance'):
    remove_edges = []
    e={}    
    for n1, n2, d in g.edges(data=True):
        e_t = []
        # Raw distance function
        if e_representation == 'chem_graph':
            if d['b_type'] is None:
                remove_edges += [(n1, n2)]
            else:
                e_t += [i+1 for i, x in enumerate([rdkit.Chem.rdchem.BondType.SINGLE, rdkit.Chem.rdchem.BondType.DOUBLE,
                                                rdkit.Chem.rdchem.BondType.TRIPLE, rdkit.Chem.rdchem.BondType.AROMATIC])
                        if x == d['b_type']]
        elif e_representation == 'distance_bin':
            if d['b_type'] is None:
                step = (6-2)/8.0
                start = 2
                b = 9
                for i in range(0, 9):
                    if d['distance'] < (start+i*step):
                        b = i
                        break
                e_t.append(b+5)
            else:
                e_t += [i+1 for i, x in enumerate([rdkit.Chem.rdchem.BondType.SINGLE, rdkit.Chem.rdchem.BondType.DOUBLE,
                                                   rdkit.Chem.rdchem.BondType.TRIPLE, rdkit.Chem.rdchem.BondType.AROMATIC])
                        if x == d['b_type']]
        elif e_representation == 'raw_distance':
            if d['b_type'] is None:
                remove_edges += [(n1, n2)]
            else:
                e_t.append(d['distance'])
                e_t += [int(d['b_type'] == x) for x in [rdkit.Chem.rdchem.BondType.SINGLE, rdkit.Chem.rdchem.BondType.DOUBLE,
                                                        rdkit.Chem.rdchem.BondType.TRIPLE, rdkit.Chem.rdchem.BondType.AROMATIC]]
        else:
            print('Incorrect Edge representation transform')
            quit()
        if e_t:
            e[(n1, n2)] = e_t
    for edg in remove_edges:
        g.remove_edge(*edg)
    return nx.to_numpy_matrix(g), e
    

def normalize_data(data, mean, std):
    data_norm = (data-mean)/std
    return data_norm


def get_values(obj, start, end, prop):
    vals = []
    for i in range(start, end):
        v = {}
        if 'degrees' in prop:
            v['degrees'] = set(sum(obj[i][0][0].sum(axis=0, dtype='int').tolist(), []))
        if 'edge_labels' in prop:
            v['edge_labels'] = set(sum(list(obj[i][0][2].values()), []))
        if 'target_mean' in prop or 'target_std' in prop:
            v['params'] = obj[i][1]
        vals.append(v)
    return vals


def get_graph_stats(graph_obj_handle, prop='degrees'):
    # if prop == 'degrees':
    num_cores = multiprocessing.cpu_count()
    inputs = [int(i*len(graph_obj_handle)/num_cores) for i in range(num_cores)] + [len(graph_obj_handle)]
    res = Parallel(n_jobs=num_cores)(delayed(get_values)(graph_obj_handle, inputs[i], inputs[i+1], prop) for i in range(num_cores))

    stat_dict = {}

    if 'degrees' in prop:
        stat_dict['degrees'] = list(set([d for core_res in res for file_res in core_res for d in file_res['degrees']]))
    if 'edge_labels' in prop:
        stat_dict['edge_labels'] = list(set([d for core_res in res for file_res in core_res for d in file_res['edge_labels']]))
    if 'target_mean' in prop or 'target_std' in prop:
        param = np.array([file_res['params'] for core_res in res for file_res in core_res])
    if 'target_mean' in prop:
        stat_dict['target_mean'] = np.mean(param, axis=0)
    if 'target_std' in prop:
        stat_dict['target_std'] = np.std(param, axis=0)

    return stat_dict


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    pred = pred.type_as(target)
    target = target.type_as(pred)
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def collate_g_concat_edge_data(batch):
    g_M = lambda g: g[0][0]
    g_n = lambda g: g_M(g).shape[0]
    g_x = lambda g: g[0][1]
    g_x_d = lambda g: len(g_x(g)[0])
    g_e = lambda g: g[0][2]
    g_m = lambda g: len(g_e(g))
    g_e_keys = lambda g: list(g_e(g).keys())
    g_e_values = lambda g: list(g_e(g).values())
    g_e_d = lambda g: len(g_e_values(g)[0])
    g_o = lambda g: g[1]
    g_o_d = lambda g: len(g[1])
    
    n_d, e_d, o_d = g_x_d(batch[0]), g_e_d(batch[0]), g_o_d(batch[0])
    N = 0
    M = 0
    batch_size = len(batch)
    for g in batch:
        n = g_n(g)
        m = g_m(g)
        N += n
        M += m
    #end for
    G = np.zeros([N, N])
    B = np.zeros([N], dtype=np.int64)
    X = np.zeros([N, n_d])
    E_d = np.zeros([2*M, e_d])
    E_src = np.zeros([2*M], dtype=np.int64)
    E_tgt = np.zeros([2*M,2], dtype=np.int64)
    Y = np.zeros([batch_size, o_d])
    
    n_acc = 0
    m_acc = 0
    for b, g in enumerate(batch):
        n = g_n(g)
        G[n_acc:n_acc+n,n_acc:n_acc+n] = g_M(g)
        B[n_acc:n_acc+n] = b
        X[n_acc:n_acc+n,:] = g_x(g)
        for edge_id, edge in enumerate(sorted(g_e_keys(g))):
            src, tgt = edge
            
            src_edge_id = m_acc+edge_id
            E_d[src_edge_id,:] = g_e(g)[edge]
            E_src[src_edge_id] = src
            E_tgt[src_edge_id,:] = [tgt,src_edge_id]
            
            tgt_edge_id = M+m_acc+edge_id
            E_d[tgt_edge_id,:] = g_e(g)[edge]
            E_src[tgt_edge_id] = tgt
            E_tgt[tgt_edge_id] = [src,tgt_edge_id]
        #end for
        Y[b] = g_o(g)
        n_acc+=n
        m_acc+=g_m(g)
    #end for

    G = torch.FloatTensor(G)
    B = torch.LongTensor(B)
    X = torch.FloatTensor(X)
    E_d = torch.FloatTensor(E_d)
    E_src = torch.LongTensor(E_src)
    E_tgt = torch.sparse.FloatTensor(torch.LongTensor(E_tgt.transpose()),torch.FloatTensor(np.ones(2*M)),torch.Size([N,2*M])).to_dense()
    Y = torch.FloatTensor(Y)
    return batch_size,G,B,X,E_d,E_src,E_tgt,Y
#end collate_g_concat

def collate_g_concat(batch):
    g_M = lambda g: g[0][0]
    g_n = lambda g: g_M(g).shape[0]
    g_x = lambda g: g[0][1]
    g_x_d = lambda g: len(g_x(g)[0])
    g_e = lambda g: g[0][2]
    g_m = lambda g: len(g_e(g))
    g_e_keys = lambda g: list(g_e(g).keys())
    g_e_values = lambda g: list(g_e(g).values())
    g_e_d = lambda g: len(g_e_values(g)[0])
    g_o = lambda g: g[1]
    g_o_d = lambda g: len(g[1])
    
    n_d, e_d, o_d = g_x_d(batch[0]), g_e_d(batch[0]), g_o_d(batch[0])
    N = 0
    M = 0
    batch_size = len(batch)
    for g in batch:
        n = g_n(g)
        m = g_m(g)
        N += n
        M += m
    #end for
    G = np.zeros([N, N])
    B = np.zeros([N], dtype=np.int64)
    X = np.zeros([N, n_d])
    E_d = np.zeros([N, N, e_d])
    E_i = np.zeros([M, 2], dtype=np.int64)
    Y = np.zeros([batch_size, o_d])
    
    n_acc = 0
    for b, g in enumerate(batch):
        n = g_n(g)
        G[n_acc:n_acc+n,n_acc:n_acc+n] = g_M(g)
        B[n_acc:n_acc+n] = b
        X[n_acc:n_acc+n,:] = g_x(g)
        for edge_id, edge in enumerate(sorted(g_e_keys(g))):
            src, tgt = edge
            E_i[edge_id,:] = [min(src,tgt),max(src,tgt)]
            E_d[n_acc+src,n_acc+tgt,:] = g_e(g)[edge]
            E_d[n_acc+tgt,n_acc+src,:] = g_e(g)[edge]
        #end for
        Y[b] = g_o(g)
        n_acc+=n
    #end for

    G = torch.FloatTensor(G)
    B = torch.LongTensor(B)
    X = torch.FloatTensor(X)
    E_d = torch.FloatTensor(E_d)
    E_i = torch.LongTensor(E_i)
    Y = torch.FloatTensor(Y)
    return G,B,X,E_d,E_i,Y
#end collate_g_concat


def collate_g_concat_dict(batch):
    g_M = lambda g: g[0][0]
    g_n = lambda g: g_M(g).shape[0]
    g_x = lambda g: g[0][1]
    g_x_d = lambda g: len(g_x(g)[0])
    g_e = lambda g: g[0][2]
    g_e_keys = lambda g: list(g_e(g).keys())
    g_e_values = lambda g: list(g_e(g).values())
    g_e_d = lambda g: len(g_e_values(g)[0])
    g_o = lambda g: g[1]
    g_o_d = lambda g: len(g[1])
    
    n_d, e_d, o_d = g_x_d(batch[0]), g_e_d(batch[0]), g_o_d(batch[0])
    N = 0
    batch_size = len(batch)
    for g in batch:
        M = g_M(g)
        n = g_n(g)
        N += n
    #end for
    G = np.zeros([N, N])
    B = np.zeros([N], dtype=np.int64)
    X = np.zeros([N, n_d])
    E = {}
    Y = np.zeros([batch_size, o_d])
    
    n_acc = 0
    print( "bla" )
    for b, g in enumerate(batch):
        n = g_n(g)
        G[n_acc:n_acc+n,n_acc:n_acc+n] = g_M(g)
        B[n_acc:n_acc+n] = b
        X[n_acc:n_acc+n,:] = g_x(g)
        for edge in g_e_keys(g):
            src,tgt=edge
            E[n_acc+src,n_acc+tgt] = g_e(g)[edge]
            E[n_acc+tgt,n_acc+src] = g_e(g)[edge]
        #end for
        Y[b] = g_o(g)
        n_acc+=n
    #end for

    G = torch.FloatTensor(G)
    B = torch.LongTensor(B)
    X = torch.FloatTensor(X)
    print( "ble" )
    for k in E.keys():
        E[k] = torch.FloatTensor(E[k])
    #end for
    Y = torch.FloatTensor(Y)
    return G,B,X,E,Y
#end collate_g_concat_dict

def collate_g(batch):

    batch_sizes = np.max(np.array([[len(input_b[1]), len(input_b[1][0]), len(input_b[2]),
                                len(list(input_b[2].values())[0])]
                                if input_b[2] else
                                [len(input_b[1]), len(input_b[1][0]), 0,0]
                                for (input_b, target_b) in batch]), axis=0)

    g = np.zeros((len(batch), batch_sizes[0], batch_sizes[0]))
    h = np.zeros((len(batch), batch_sizes[0], batch_sizes[1]))
    e = np.zeros((len(batch), batch_sizes[0], batch_sizes[0], batch_sizes[3]))

    target = np.zeros((len(batch), len(batch[0][1])))

    for i in range(len(batch)):

        num_nodes = len(batch[i][0][1])

        # Adjacency matrix
        g[i, 0:num_nodes, 0:num_nodes] = batch[i][0][0]

        # Node features
        h[i, 0:num_nodes, :] = batch[i][0][1]

        # Edges
        for edge in batch[i][0][2].keys():
            e[i, edge[0], edge[1], :] = batch[i][0][2][edge]
            e[i, edge[1], edge[0], :] = batch[i][0][2][edge]

        # Target
        target[i, :] = batch[i][1]

    g = torch.FloatTensor(g)
    h = torch.FloatTensor(h)
    e = torch.FloatTensor(e)
    target = torch.FloatTensor(target)

    return g, h, e, target


def save_checkpoint(state, is_best, directory):

    if not os.path.isdir(directory):
        os.makedirs(directory)
    checkpoint_file = os.path.join(directory, 'checkpoint.pth')
    best_model_file = os.path.join(directory, 'model_best.pth')
    torch.save(state, checkpoint_file)
    if is_best:
        shutil.copyfile(checkpoint_file, best_model_file)


