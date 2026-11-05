import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import sqlite3
import csv
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset,Dataset
from sklearn.model_selection import StratifiedKFold
# from sklearn.metrics import accuracy_score
# from pytorchtools import EarlyStopping
from modeltask1 import FusionLayer,GNN1,GNN2,GNN3,GNN4
from collections import defaultdict
import os
import random

from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score

import warnings

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='GNN based on the whole datas')
parser.add_argument("--epoches",type=int,default=120)
parser.add_argument("--batch_size",type=int,choices=[2048,1024,512,256,128],default=1024)
parser.add_argument("--weigh_decay",type=float,choices=[1e-1,1e-2,1e-3,1e-4,1e-8],default=1e-8)
parser.add_argument("--lr",type=float,choices=[1e-3,1e-4,1e-5,4*1e-3],default=1*1e-2) #4*1e-3

parser.add_argument("--neighbor_sample_size",choices=[4,6,10,16],type=int,default=6)
parser.add_argument("--event_num",type=int,default=65)

parser.add_argument("--n_drug",type=int,default=572)
parser.add_argument("--seed",type=int,default=0)
parser.add_argument("--dropout",type=float,default=0.3)
parser.add_argument("--embedding_num",type=int,choices=[128,64,256,32],default=128)
# Ablation: bỏ 1 knowledge graph khỏi fusion (0=đủ, 1..4 = bỏ G1..G4)
parser.add_argument("--drop_kg",type=int,default=0)
# Weights & Biases (optional, off by default so existing runs are unchanged)
parser.add_argument("--use_wandb",type=int,default=0)
parser.add_argument("--wandb_project",type=str,default="gml-mkg-fenn")
parser.add_argument("--wandb_name",type=str,default="")
parser.add_argument("--wandb_entity",type=str,default="")
parser.add_argument("--wandb_group",type=str,default="")
parser.add_argument("--wandb_notes",type=str,default="")
args = parser.parse_args()

# cùng tên chỉ số với các run W&B đã có của nhóm
_METRIC_NAMES = ["ACC","AUPR_micro","AUPR_macro","AUC_micro","AUC_macro",
                 "F1_micro","F1_macro","Pre_micro","Pre_macro",
                 "Rec_micro","Rec_macro"]

# GPU neu co (code goc chi CPU; them .to(_DEVICE) de chay GPU nhanh hon ~nhieu)
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_seed():
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

def read_dataset(drug_name_id,num):
    kg = defaultdict(list)
    tails = {}
    relations = {}
    drug_list=[]
    filename = "../dataset/dataset"+str(num)+".txt"
    with open(filename, encoding="utf8") as reader:
        for line in reader:
            string= line.rstrip().split('//',2)
            head=string[0]
            tail=string[1]
            relation=string[2]
            drug_list.append(drug_name_id[head])
            if tail not in tails:
                tails[tail] = len(tails)
            if relation not in relations:
                relations[relation] = len(relations)
            if num==3:
                kg[drug_name_id[head]].append((drug_name_id[tail], relations[relation]))
                kg[drug_name_id[tail]].append((drug_name_id[head], relations[relation]))
            else:
                kg[drug_name_id[head]].append((tails[tail], relations[relation]))
    return kg,len(tails),len(relations)


def prepare(mechanism, action):
    d_label = {}
    d_event = []
    new_label = []
    for i in range(len(mechanism)):
        d_event.append(mechanism[i] + " " + action[i])
    count = {}
    for i in d_event:
        if i in count:
            count[i] += 1
        else:
            count[i] = 1
    list1 = sorted(count.items(), key=lambda x: x[1], reverse=True)
    for i in range(len(list1)):
        d_label[list1[i][0]] = i
    for i in range(len(d_event)):
        new_label.append(d_label[d_event[i]])
    return new_label

def l2_re(parameter):
    reg=0
    for param in parameter:
        reg+=0.5*(param**2).sum()
    return reg

def roc_aupr_score(y_true, y_score, average="macro"):
    def _binary_roc_aupr_score(y_true, y_score):
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)
        return auc(recall, precision)
    def _average_binary_score(binary_metric, y_true, y_score, average):  # y_true= y_one_hot
        if average == "binary":
            return binary_metric(y_true, y_score)
        if average == "micro":
            y_true = y_true.ravel()
            y_score = y_score.ravel()
        if y_true.ndim == 1:
            y_true = y_true.reshape((-1, 1))
        if y_score.ndim == 1:
            y_score = y_score.reshape((-1, 1))
        n_classes = y_score.shape[1]
        score = np.zeros((n_classes,))
        for c in range(n_classes):
            y_true_c = y_true.take([c], axis=1).ravel()
            y_score_c = y_score.take([c], axis=1).ravel()
            score[c] = binary_metric(y_true_c, y_score_c)
        return np.average(score)
    return _average_binary_score(_binary_roc_aupr_score, y_true, y_score, average)

def evaluate(pred_type, pred_score, y_test, event_num):
    all_eval_type = 11
    result_all = np.zeros((all_eval_type, 1), dtype=float)
    each_eval_type = 6
    result_eve = np.zeros((event_num, each_eval_type), dtype=float)
    # label_binarize:返回一个one_hot的类型
    y_one_hot = label_binarize(y_test, classes=np.arange(event_num))
    pred_one_hot = label_binarize(pred_type,classes= np.arange(event_num))
    result_all[0] = accuracy_score(y_test, pred_type)
    result_all[1] = roc_aupr_score(y_one_hot, pred_score, average='micro')
    result_all[2] = roc_aupr_score(y_one_hot, pred_score, average='macro')
    result_all[3] = roc_auc_score(y_one_hot, pred_score, average='micro')
    result_all[4] = roc_auc_score(y_one_hot, pred_score, average='macro')
    result_all[5] = f1_score(y_test, pred_type, average='micro')
    result_all[6] = f1_score(y_test, pred_type, average='macro')
    result_all[7] = precision_score(y_test, pred_type, average='micro')
    result_all[8] = precision_score(y_test, pred_type, average='macro')
    result_all[9] = recall_score(y_test, pred_type, average='micro')
    result_all[10] = recall_score(y_test, pred_type, average='macro')
    for i in range(event_num):
        result_eve[i, 0] = accuracy_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel())
        result_eve[i, 1] = roc_aupr_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),
                                          average=None)
        result_eve[i, 2] = roc_auc_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),
                                         average=None)
        result_eve[i, 3] = f1_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),
                                    average='binary')
        result_eve[i, 4] = precision_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),
                                           average='binary')
        result_eve[i, 5] = recall_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),
                                        average='binary')
    return [result_all, result_eve]


def save_result(filepath,result_type,result):
    with open(filepath+result_type +'task1'+'.csv', "w", newline='',encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        for i in result:
            writer.writerow(i)
    return 0

def train(train_x,train_y,test_x,test_y,net,fold=0):
    loss_function=nn.CrossEntropyLoss()
    opti = torch.optim.Adam(net.parameters(), lr=args.lr,weight_decay=args.weigh_decay)
    test_loss, test_acc, train_l = 0, 0, 0
    train_a = []
    train_x1 = train_x.copy()
    train_x[:,[0,1]] = train_x[:,[1,0]]
    train_x_total = torch.LongTensor(np.concatenate([train_x1, train_x], axis=0))
    train_y = torch.LongTensor(np.concatenate([train_y,train_y]))
    train_data = TensorDataset(train_x_total, train_y)
    train_iter = DataLoader(train_data, args.batch_size, shuffle=True)
    test_list = []
    max_test_output = torch.zeros((0,65),dtype=torch.float)
    for epoch in range(args.epoches):
        test_loss, test_score, train_l = 0, 0, 0
        train_a = []
        net.train()
        for x, y in train_iter:
            opti.zero_grad()
            train_acc = 0
            train_label = torch.LongTensor(y).to(_DEVICE)
            x = torch.LongTensor(x).to(_DEVICE)
            output = net(x)
            l = loss_function(output, train_label)
            l.backward()
            opti.step()
            train_l += l.item()
            train_acc = accuracy_score(torch.argmax(output,dim=1).cpu(), train_label.cpu())
            train_a.append(train_acc)
        net.eval()
        with torch.no_grad():
            test_x_t = torch.LongTensor(test_x).to(_DEVICE)
            test_output = F.softmax(net(test_x_t),dim=1)
            test_label = torch.LongTensor(test_y).to(_DEVICE)
            loss = loss_function(test_output, test_label)
            test_loss = loss.item()
            _pred = torch.argmax(test_output,dim=1).cpu()
            _yl = test_label.cpu()
            test_score = f1_score(_pred, _yl, average='macro')
            test_acc = accuracy_score(_pred, _yl)
            test_list.append(test_score)
            if test_score==max(test_list):
                max_test_output = test_output.cpu()
            print("test_acc:", test_acc, "train_acc:", sum(train_a) / len(train_a),"test_score:", test_score)
            if args.use_wandb:
                import wandb
                # ghi trong lúc train: loss và chỉ số test sau mỗi epoch của từng fold
                wandb.log({"fold": fold, "epoch": epoch + 1,
                           "global_epoch": fold * args.epoches + epoch + 1,
                           "train_loss": train_l / len(train_y), "test_loss": test_loss,
                           "train_acc": sum(train_a) / len(train_a),
                           "test_acc": test_acc, "test_f1_macro": test_score})
        print('epoch [%d] train_loss: %.6f testing_loss: %.6f ' % (
                epoch + 1, train_l / len(train_y), test_loss / len(test_y)))
    return test_loss / len(test_y), max(test_list), train_l / len(train_y), sum(train_a) / len(
        train_a), test_list, max_test_output



def main():
    if args.use_wandb:
        import wandb
        wandb.init(entity=(args.wandb_entity or None), project=args.wandb_project,
                   group=(args.wandb_group or None), notes=(args.wandb_notes or None),
                   name=(args.wandb_name or ("task1_seed%d" % args.seed)),
                   config={k: v for k, v in vars(args).items() if not k.startswith("wandb")},
                   # không gửi metadata máy (tên máy, đường dẫn, requirements) lên W&B
                   settings=wandb.Settings(x_disable_meta=True, x_disable_stats=True,
                                    x_disable_machine_info=True, x_save_requirements=False,
                                    disable_git=True, save_code=False, disable_code=True,
                                    disable_job_creation=True, host="gpu-node"))
    conn = sqlite3.connect("../dataset/event.db")
    df_drug = pd.read_sql('select * from drug;', conn)
    extraction = pd.read_sql('select * from extraction;', conn)
    mechanism = extraction['mechanism']
    action = extraction['action']
    drugA = extraction['drugA']
    drugB = extraction['drugB']
    new_label = prepare(mechanism, action)
    new_label = np.array(new_label)
    dict1 = {}
    for i in df_drug["name"]:
        dict1[i] = len(dict1)
    drug_name = [dict1[i] for i in df_drug["name"]]
    drugA_id = [dict1[i] for i in drugA]
    drugB_id = [dict1[i] for i in drugB]
    dataset1_kg, dataset1_tail_len, dataset1_relation_len = read_dataset(dict1,1)
    dataset2_kg, dataset2_tail_len, dataset2_relation_len = read_dataset(dict1,2)
    dataset3_kg, dataset3_tail_len, dataset3_relation_len = read_dataset(dict1,3)
    dataset4_kg, dataset4_tail_len, dataset4_relation_len = read_dataset(dict1, 4)
    x_datasets = {"drugA": drugA_id, "drugB": drugB_id}
    x_datasets = pd.DataFrame(data=x_datasets)
    x_datasets = x_datasets.to_numpy()
    dataset={}
    dataset["dataset1"],dataset["dataset2"],dataset["dataset3"],dataset["dataset4"] = dataset1_kg,dataset2_kg,dataset3_kg,dataset4_kg
    tail_len={}
    tail_len["dataset1"],tail_len["dataset2"],tail_len["dataset3"],tail_len["dataset4"] = dataset1_tail_len,dataset2_tail_len,dataset3_tail_len,dataset4_tail_len
    relation_len={}
    relation_len["dataset1"],relation_len["dataset2"],relation_len["dataset3"],relation_len["dataset4"] = dataset1_relation_len,dataset2_relation_len,dataset3_relation_len,dataset4_relation_len
    train_sum, test_sum = 0, 0

    y_true = np.array([])
    y_score = np.zeros((0, 65), dtype=float)
    y_pred = np.array([])

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

    kfold = kf.split(x_datasets, new_label)
    for i, (train_idx, test_idx) in enumerate(kfold):
        net = nn.Sequential(GNN1(dataset, tail_len, relation_len, args, dict1, drug_name),
                            GNN2(dataset, tail_len, relation_len, args, dict1, drug_name),
                            GNN3(dataset, tail_len, relation_len, args, dict1, drug_name),
                            GNN4(dataset, tail_len, relation_len, args, dict1, drug_name),
                            FusionLayer(args))
        net = net.to(_DEVICE)
        train_x = x_datasets[train_idx]
        train_y = new_label[train_idx]
        test_x = x_datasets[test_idx]
        test_y = new_label[test_idx]
        test_loss, test_acc, train_loss, train_acc, test_list,test_output = train(train_x,train_y,test_x,test_y,net,fold=i)
        train_sum += train_acc
        test_sum += test_acc
        pred_type = torch.argmax(test_output, dim=1).numpy()
        y_pred = np.hstack((y_pred, pred_type))
        y_score = np.row_stack((y_score, test_output))
        y_true = np.hstack((y_true, test_y))
        print('fold %d, test_loss %f, test_acc %f, train_loss %f, train_acc %f' % (
            i, test_loss, test_acc, train_loss, train_acc))
        if args.use_wandb:
            # 'test_acc' trả về từ train() thực ra là macro-F1 test cao nhất của fold (giao thức gốc
            # chọn epoch theo tập test); ghi kèm epoch được chọn và F1 ở epoch cuối để so sánh
            wandb.log({"fold": i, "fold_selected_f1_macro": test_acc,
                       "fold_selected_epoch": int(np.argmax(test_list)) + 1,
                       "fold_last_f1_macro": test_list[-1]})
    result_all, result_eve = evaluate(y_pred, y_score, y_true, args.event_num)
    save_result("../result/", "all", result_all)
    save_result("../result/", "each", result_eve)
    print('%d-fold validation: avg train acc  %f, avg test acc %f' % (i, train_sum / 5, test_sum / 5))
    if args.use_wandb:
        final = {_METRIC_NAMES[k]: float(result_all[k][0]) for k in range(len(_METRIC_NAMES))}
        final["avg_train_acc"] = train_sum / 5
        final["avg_test_acc"] = test_sum / 5
        wandb.log(final)
        wandb.summary.update(final)
        wandb.finish()
    return

if __name__ == '__main__':
    setup_seed()
    main()