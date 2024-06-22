import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch_geometric.utils import from_networkx
import argparse
import os
import random
import numpy as np
import pandas as pd
import networkx as nx
import scipy
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, recall_score, f1_score
import faulthandler   
from datetime import datetime

from models import MDMNI_DGD
from network import generate_graph
from utils import *

cuda = torch.cuda.is_available()
torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def seed_torch(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description="Case study……")
    parser.add_argument('--networks', nargs='+', type=int, default=[1, 2, 3, 4, 5, 6], help='Indices of networks to use')
    parser.add_argument('--thr_go', dest='thr_go', default=0.8, type=float, help='the threshold for GO semantic similarity')
    parser.add_argument('--thr_seq', dest='thr_seq', default=0.8, type=float, help='the threshold for sequence similarity')
    parser.add_argument('--thr_exp', dest='thr_exp', default=0.8, type=float, help='the threshold for gene co-expression pattern')
    parser.add_argument('--thr_path', dest='thr_path', default=0.6, type=float, help='the threshold of pathway co-occurrence')
    parser.add_argument('--thr_cpdb', dest='thr_cpdb', default=0.8, type=float, help='the threshold of CPDB PPI')
    parser.add_argument('--thr_string', dest='thr_string', default=0.8, type=float, help='the threshold of STRING PPI')
    parser.add_argument('--thr_domain', dest='thr_domain', default=0.3, type=float, help='the threshold of Domain similarity')
    parser.add_argument('--epochs', dest='epochs', default=1000, type=int, help='maximum number of epochs')
    parser.add_argument('--patience', dest='patience', default=120, type=int, help='waiting iterations when performance no longer improves')
    parser.add_argument('--hidden_dim', dest='hidden_dim', default=256, type=int, help='the dim of hidden layer')
    parser.add_argument('--output_dim', dest='output_dim', default=32, type=int, help='the dim of output')
    parser.add_argument('--dropout', dest='dropout', default=0.2, type=float, help='the dropout rate')
    parser.add_argument('--lr', dest='lr', default=0.001, type=float, help='the learning rate')
    parser.add_argument('--wd', dest='wd', default=0.0005, type=float, help='the weight decay')
    parser.add_argument('--seed', default=42, type=int, help='the random seed')
    parser.add_argument('--k', default=10, type=int, help='KFold')
    parser.add_argument('--train_name', default='All_genes', type=str, help='the name of feature file')
    args = parser.parse_args()
    return args


def main(args):
    seed_torch(args['seed'])

    graph_path = os.path.join('Data/network')
    if not os.path.exists(graph_path):
        os.makedirs(graph_path)
    file_save_path = os.path.join('results')
    feature_path = os.path.join('Data/feature')
    train_path = os.path.join(feature_path, str(args['train_name']) + '.csv')
    train_file = pd.read_csv(train_path)                                             
    #final_gene_node = train_file['gene'].tolist()
    train_data = train_file.iloc[:, 1:-1]                                            
    input_dim = train_data.shape[1]                                                  

    generate_input = generate_graph(graph_path)

    print("\n* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * ")
    print('Loading train network datasets……')
    cpdb_path = os.path.join(graph_path, str(args['train_name'])+'_cpdb_' + str(args['thr_cpdb']) + '.csv')
    string_path = os.path.join(graph_path, str(args['train_name'])+'_string_' + str(args['thr_string']) + '.csv')
    exp_path = os.path.join(graph_path, str(args['train_name'])+'_exp_' + str(args['thr_exp']) + '.csv')
    go_path = os.path.join(graph_path, str(args['train_name'])+'_go_' + str(args['thr_go']) + '.csv')
    path_path = os.path.join(graph_path, str(args['train_name'])+'_path_' + str(args['thr_path']) + '.csv')
    seq_path = os.path.join(graph_path, str(args['train_name'])+'_seq_' + str(args['thr_seq']) + '.csv')
    dom_path = os.path.join(graph_path, str(args['train_name'])+'_domain_' + str(args['thr_domain']) + '.csv')

    if os.path.exists(cpdb_path) & os.path.exists(string_path) & os.path.exists(dom_path) & os.path.exists(go_path) & os.path.exists(exp_path) & os.path.exists(seq_path) & os.path.exists(path_path):
        print('The train profiles already exist!\n')
        cpdb_file = pd.read_csv(cpdb_path, index_col=0)
        string_file = pd.read_csv(string_path, index_col=0)
        exp_file = pd.read_csv(exp_path, index_col=0)
        go_file = pd.read_csv(go_path, index_col=0)
        path_file = pd.read_csv(path_path, index_col=0)
        seq_file = pd.read_csv(seq_path, index_col=0)
        dom_file = pd.read_csv(dom_path, index_col=0)
    else:
        cpdb_file, string_file, dom_file, exp_file, go_file, path_file, seq_file = generate_input.generate_graph(
               train_path, args['train_name'], args['thr_cpdb'], args['thr_string'], args['thr_domain'],
                           args['thr_exp'], args['thr_go'], args['thr_path'], args['thr_seq'])


    print("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * ")

    print('Selected Network INFO：')
    name_of_network = ['CPDB', 'STRING', 'DOMAIN', 'EXP', 'GO', 'PATH', 'SEQ']
    graphlist = []
    for i, network in enumerate([cpdb_file, string_file, dom_file, exp_file, go_file, path_file, seq_file]):
        featured_graph = load_featured_graph(network, train_data)                
        graphlist.append(featured_graph)

    selected_networks = [graphlist[i] for i in args['networks']]                 
    for sel_i in args['networks']:
        print(f"    The {name_of_network[sel_i]} network.")
    graphlist_adj = [graph.cuda() for graph in selected_networks]                

    print("\n* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * ")

    def train(mask, label):
        model.train()
        optimizer.zero_grad()
        output = model(graphlist_adj)
        loss = F.binary_cross_entropy_with_logits(output[mask], label, pos_weight=torch.Tensor([1]).cuda())
        acc = metrics.accuracy_score(label.cpu(), np.round(torch.sigmoid(output[mask]).cpu().detach().numpy()))
        loss.backward()
        optimizer.step()
        del output
        return loss.item(), acc

    @torch.no_grad()
    def test(mask, label):
        model.eval()
        output = model(graphlist_adj)
        loss = F.binary_cross_entropy_with_logits(output[mask], label, pos_weight=torch.Tensor([1]).cuda())
        acc = metrics.accuracy_score(label.cpu(), np.round(torch.sigmoid(output[mask]).cpu().detach().numpy()))
        pred = torch.sigmoid(output[mask]).cpu().detach().numpy()
        auroc = metrics.roc_auc_score(label.to('cpu'), pred)
        pr, rec, _ = metrics.precision_recall_curve(label.to('cpu'), pred)
        aupr = metrics.auc(rec, pr)
        return pred, loss.item(), acc, auroc, aupr

#-----------------------------------------------------------------------------------------------------------------------

    train_file = pd.read_csv(train_path)
    idx_list = train_file.index.tolist()
    label_list = train_file['label'].tolist()
    print(f'The number of gene index: {len(idx_list)}')

    # Case studying on 20,544 genes
    print("\n* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * ")
    print("Case Study……")
    best_model_state_dict = torch.load('MDMNI-DGD_best_model.pt')

    all_mask = torch.LongTensor(idx_list)
    all_label = torch.FloatTensor(label_list).reshape(-1, 1)

    model = MDMNI_DGD(input_dim, hidden_dim=args['hidden_dim'], output_dim=args['output_dim'], dropout=args['dropout'])
    model.cuda()
    model.load_state_dict(best_model_state_dict)

    all_pred, _, _, _, _ = test(all_mask.cuda(), all_label.cuda())

    all_genes = train_file.loc[idx_list, 'gene']
    all_labels = all_label.flatten()
    all_preds = all_pred.flatten()

    df = pd.DataFrame({'gene': all_genes, 'label': all_labels, 'prediction': all_preds})
    df.to_csv('results/Case_study_result.csv', index=False)
    print("Saved it successfully！")



if __name__ == '__main__':
    args = parse_args()
    args_dic = vars(args)
    print('\n* args_dict：\n', args_dic)
    main(args_dic)
    print('The case study is finished!\n')