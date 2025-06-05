import torch
import os
import argparse
import random
from joblib import dump, load
import numpy as np
from torch.utils.data import DataLoader
from utils import *
from data.Task import *
from models.Model import *
from models.baselines import *
torch.distributed.init_process_group(backend="nccl")

# print(os.environ['LOCAL_RANK'])
local_rank = int(os.environ['LOCAL_RANK'])
# local_rank = args.local_rank
# local_rank = rank
torch.cuda.set_device(local_rank)
# global device
device = torch.device("cuda", local_rank)

fileroot = {
   'mimic3': '/home/lzh/dataset/mimic3',
   'mimic4': 'data path of mimic4',
   'ccae': './data/processed_dip.pkl'
}

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100, help = 'Number of epochs to train.')
parser.add_argument('--lr', type=float, default = 0.001, help = 'learning rate.')
parser.add_argument('--model', type=str, default="TRANS", help = 'Transformer, RETAIN, StageNet, KAME, GCT, DDHGNN, TRANS')
parser.add_argument('--dev', type=int, default = 2)
parser.add_argument('--seed', type=int, default = 42)
parser.add_argument('--dataset', type=str, default = "mimic3", choices=['mimic3', 'mimic4', 'ccae'])
parser.add_argument('--batch_size', type=int, default = 128)
parser.add_argument('--pe_dim', type=int, default = 4, help = 'dimensions of spatial encoding')
parser.add_argument('--devm', type=bool, default = False, help = 'develop mode')
parser.add_argument('--device', type=str, default="cuda:1", help='GPU device')
parser.add_argument("--local_rank", type=int, default=0, help="number of cpu threads to use during batch generation")


args = parser.parse_args()
# device = torch.device("cuda", local_rank)
# device = 'cuda:1'
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

if args.dataset == 'mimic4':
    task_dataset = load_dataset(args.dataset, root=fileroot[args.dataset], task_fn=diag_prediction_mimic4_fn,
                                dev=args.devm)
elif args.dataset == 'mimic3':
    task_dataset = load_dataset(args.dataset, root=fileroot[args.dataset], task_fn=diag_prediction_mimic3_fn,
                                dev=args.devm)
else:
    task_dataset = load_dataset(args.dataset, root=fileroot[args.dataset])

Tokenizers = get_init_tokenizers(task_dataset)
label_tokenizer = Tokenizer(tokens=task_dataset.get_all_tokens('conditions'))
if args.model == 'Transformer':
    train_loader, val_loader, test_loader = seq_dataloader(task_dataset, batch_size=args.batch_size)
    model = Transformer(Tokenizers, len(task_dataset.get_all_tokens('conditions')), device)

elif args.model == 'RETAIN':
    train_loader, val_loader, test_loader = seq_dataloader(task_dataset, batch_size=args.batch_size)
    model = RETAIN(Tokenizers, len(task_dataset.get_all_tokens('conditions')), device)

elif args.model == 'KAME':
    train_loader, val_loader, test_loader = seq_dataloader(task_dataset, batch_size=args.batch_size)
    Tokenizers.update(get_parent_tokenizers(task_dataset))
    model = KAME(Tokenizers, len(task_dataset.get_all_tokens('conditions')), device)

elif args.model == 'StageNet':
    train_loader, val_loader, test_loader = seq_dataloader(task_dataset, batch_size=args.batch_size)
    model = StageNet(Tokenizers, len(task_dataset.get_all_tokens('conditions')), device)

elif args.model == 'TRANS':
    data_path = './logs/{}_{}_16_new.pkl'.format(args.dataset, args.pe_dim)
    # data_path = './logs/{}_{}_16.pkl'.format(args.dataset, args.pe_dim)
    if os.path.exists(data_path):
        mdataset = load(data_path)
    else:
        mdataset = MMDataset(task_dataset, Tokenizers, dim=128, device=device, trans_dim=args.pe_dim)
        dump(mdataset, data_path)
    trainset, validset, testset = split_dataset(mdataset)
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
    valid_sampler = torch.utils.data.distributed.DistributedSampler(validset)
    test_sampler = torch.utils.data.distributed.DistributedSampler(testset)
    train_loader = DataLoader(trainset, batch_size=args.batch_size, collate_fn=custom_collate_fn, sampler=train_sampler)
    valid_loader = DataLoader(trainset, batch_size=args.batch_size, collate_fn=custom_collate_fn, sampler=valid_sampler)
    test_loader = DataLoader(trainset, batch_size=args.batch_size, collate_fn=custom_collate_fn, sampler=test_sampler)
    model = TRANS(Tokenizers, 128, len(task_dataset.get_all_tokens('conditions')),
                  device, graph_meta=graph_meta, pe=args.pe_dim)

ckptpath = './logs/trained_{}_{}.ckpt'.format(args.model, args.dataset)
# best_model_0 = torch.load('trained_TRANS_mimic3_0.ckpt')
# best_model_1 = torch.load('trained_TRANS_mimic3_1.ckpt')

# local_rank = torch.distributed.get_rank()
model.to(device)
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank,
                                                      find_unused_parameters=True)

if local_rank == 0:
    best_model = torch.load(ckptpath)
    # new_state_dict = {}
    # for key, value in best_model.items():
    #     new_key = key
    #     while new_key.startswith('module.'):
    #         new_key = new_key[len('module.'):]
    #     new_key = 'module.'+new_key if 'module.' not in new_key else new_key
    #     new_state_dict[new_key] = value
    # model.load_state_dict(new_state_dict)
    model.module.load_state_dict(best_model)
    model = model.to(device)

    y_t_all, y_p_all = [], []
    y_true, y_prob = test(test_loader, model, label_tokenizer)
    print(code_level(y_true, y_prob))
    print(visit_level(y_true, y_prob))