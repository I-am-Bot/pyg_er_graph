from torch_geometric.data import Data, DataLoader
import argparse
import pickle as cp

cmd_opt = argparse.ArgumentParser(description='Argparser for molecule vae')
cmd_opt.add_argument('-data_folder', type=str, default=None, help='data folder')
cmd_opt.add_argument('-min_n', type=int, default=0, help='min #nodes')
cmd_opt.add_argument('-max_n', type=int, default=0, help='max #nodes')
cmd_opt.add_argument('-er_p', type=float, default=0, help='parameter of er graphs')
cmd_opt.add_argument('-n_graphs', type=int, default=0, help='number of graphs')
cmd_args, _ = cmd_opt.parse_known_args()


def load_dataset(cmd_args):
    pattern = 'nrange-%d-%d-n_graph-%d-p-%.2f' % (cmd_args.min_n, cmd_args.max_n, cmd_args.n_graphs, cmd_args.er_p)
    with open('%s/%s_train.pkl' % (cmd_args.data_folder, pattern), 'rb') as f:
        train_glist = cp.load(f)
    with open('%s/%s_test.pkl' % (cmd_args.data_folder, pattern), 'rb') as f:
        test_glist = cp.load(f)
    train_loader = DataLoader(train_glist)
    test_loader = DataLoader(test_glist)
    return train_loader, test_loader

train_loader, test_loader = load_dataset(cmd_args)

for graph in train_loader:
    # fill your code here
    print('Number of nodes:', graph.num_nodes)
