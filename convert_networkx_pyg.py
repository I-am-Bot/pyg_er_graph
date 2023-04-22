from __future__ import print_function

import os
import sys
import numpy as np
import torch
import networkx as nx
import random
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from cmd_args import cmd_args
from graph_embedding import S2VGraph
import pickle
import json
from copy import deepcopy

def load_pkl(fname, num_graph):
    g_list = []
    with open(fname, 'rb') as f:
        for i in range(num_graph):
            g = cp.load(f)
            g_list.append(g)
    return g_list

def networkx_to_json():
    import ipdb
    ipdb.set_trace()
    frac_train = 0.9
    pattern = 'nrange-%d-%d-n_graph-%d-p-%.2f' % (cmd_args.min_n, cmd_args.max_n, cmd_args.n_graphs, cmd_args.er_p)

    num_train = int(frac_train * cmd_args.n_graphs)

    graph_dicts_test = []
    graph_dicts_train = []
    label_map = {}

    for i in range(cmd_args.min_c, cmd_args.max_c + 1):
        cur_list = load_pkl('%s/ncomp-%d-%s.pkl' % (cmd_args.data_folder, i, pattern), cmd_args.n_graphs)
        assert len(cur_list) == cmd_args.n_graphs
        for j in range(num_train):
            graph_embedding = S2VGraph(cur_list[j], i).__dict__
            graph_embedding.pop('edge_pairs')
            graph_dicts_train.append(graph_embedding)
        for j in range(num_train, len(cur_list)):
            graph_embedding = S2VGraph(cur_list[j], i).__dict__
            graph_embedding.pop('edge_pairs')
            graph_dicts_test.append(graph_embedding)

        label_map[i] = i - cmd_args.min_c
    cmd_args.num_class = len(label_map)
    cmd_args.feat_dim = 1

    def encode_array(obj):
        if isinstance(obj, list):
            return {'__type__': 'list', 'data': obj}
        else:
            return obj
    json_str_train = json.dumps(graph_dicts_train, default=encode_array)
    
    with open('%s_train.json' % pattern, 'w') as f:
        f.write(json_str_train)
        print('train set')

    json_str_test = json.dumps(graph_dicts_test, default=encode_array)

    with open('%s_test.json' % pattern, 'w') as f:
        f.write(json_str_test)
        print('test set')
    return label_map, graph_dicts_train, graph_dicts_test

def json_to_pyg():
    pattern = 'nrange-%d-%d-n_graph-%d-p-%.2f' % (cmd_args.min_n, cmd_args.max_n, cmd_args.n_graphs, cmd_args.er_p)
    import ipdb
    ipdb.set_trace()
    from torch_geometric.data import Data
    for i in range(cmd_args.min_c, cmd_args.max_c + 1):

        train_glist = []
        test_glist = []
        def decode_array(obj):
            if '__type__' in obj and obj['__type__'] == 'list':
                return obj['data']
            else:
                return obj
        with open('%s_train.json' % pattern, 'r') as f:
            graph_dicts_train = json.load(f, object_hook=decode_array)
        with open('%s_test.json' % pattern, 'r') as f:
            graph_dicts_test = json.load(f, object_hook=decode_array)
        
        for graph in graph_dicts_train:
            data = Data(x=torch.zeros((graph['num_nodes'], 0)), edge_index=torch.tensor(graph['myedges']))
            data.label = torch.tensor([graph['label']])
            train_glist.append(data)
        
        for graph in graph_dicts_test:
            data = Data(x=torch.zeros((graph['num_nodes'], 0)), edge_index=torch.tensor(graph['myedges']))
            data.label = torch.tensor([graph['label']])
            test_glist.append(data)
    with open('%s_train.pkl' % pattern, 'wb') as f:
        pickle.dump(train_glist, f)
    with open('%s_test.pkl' % pattern, 'wb') as f:
        pickle.dump(test_glist, f)
    return train_glist, test_glist

def networkx_to_pyg():
    from torch_geometric.data import Data
    frac_train = 0.9
    pattern = 'nrange-%d-%d-n_graph-%d-p-%.2f' % (cmd_args.min_n, cmd_args.max_n, cmd_args.n_graphs, cmd_args.er_p)

    num_train = int(frac_train * cmd_args.n_graphs)

    train_glist = []
    test_glist = []
    label_map = {}

    for i in range(cmd_args.min_c, cmd_args.max_c + 1):
        cur_list = load_pkl('%s/ncomp-%d-%s.pkl' % (cmd_args.data_folder, i, pattern), cmd_args.n_graphs)
        assert len(cur_list) == cmd_args.n_graphs

        for j in range(num_train):
            graph_embedding = S2VGraph(cur_list[j], i).__dict__
            data = Data(x=torch.zeros((graph_embedding['num_nodes'], 0)), edge_index=torch.tensor(graph_embedding['myedges']))
            data.label = torch.tensor([graph_embedding['label']])
            train_glist.append(data)
        
        for j in range(num_train, len(cur_list)):
            graph_embedding = S2VGraph(cur_list[j], i).__dict__
            data = Data(x=torch.zeros((graph_embedding['num_nodes'], 0)), edge_index=torch.tensor(graph_embedding['myedges']))
            data.label = torch.tensor([graph_embedding['label']])
            test_glist.append(data)

        label_map[i] = i - cmd_args.min_c
    cmd_args.num_class = len(label_map)
    cmd_args.feat_dim = 1
    print('# train:', len(train_glist), ' # test:', len(test_glist))
    with open('%s_train.pkl' % pattern, 'wb') as f:
        pickle.dump(train_glist, f)
    with open('%s_test.pkl' % pattern, 'wb') as f:
        pickle.dump(test_glist, f)
    return label_map, train_glist, test_glist