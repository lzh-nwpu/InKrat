import os
from tqdm import *
import random
import argparse
import numpy as np
from joblib import dump, load
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import *
from data.Task import *
from models.Model import *
from models.baselines import *
# from torch.cuda.amp import GradScaler
torch.distributed.init_process_group(backend="nccl")



parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate.')
parser.add_argument('--model', type=str, default="TRANS",
                    help='Transformer, RETAIN, StageNet, KAME, GCT, DDHGNN, TRANS')
parser.add_argument('--dev', type=int, default=7)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--dataset', type=str, default="mimic3", choices=['mimic3', 'mimic4', 'ccae'])
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--pe_dim', type=int, default=4, help='dimensions of spatial encoding')
parser.add_argument('--devm', type=bool, default=False, help='develop mode')
# parser.add_argument("--local_rank", type=int, default=0, help="number of cpu threads to use during batch generation")


fileroot = {
    'mimic3': '/home/lzh/dataset/mimic3',
    'mimic4': 'data path of mimic4',
    'ccae': './data/processed_dip.pkl'
}

args = parser.parse_args()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
print('{}--{}'.format(args.dataset, args.model))
# cudaid = "cuda:" + str(args.dev)
# device = torch.device(cudaid if torch.cuda.is_available() else "cpu")
# print(os.environ['LOCAL_RANK'])
# local_rank = torch.distributed.get_rank()
local_rank = int(os.environ['LOCAL_RANK'])
# local_rank = args.local_rank
# local_rank = rank
torch.cuda.set_device(local_rank)
# global device
device = torch.device("cuda", local_rank)
print(torch.cuda.current_device())
# print(local_rank, device)

# if args.dataset == 'mimic4':
#     task_dataset = load_dataset(args.dataset, root=fileroot[args.dataset], task_fn=diag_prediction_mimic4_fn,
#                                 dev=args.devm)
# elif args.dataset == 'mimic3':
#     task_dataset = load_dataset(args.dataset, root=fileroot[args.dataset], task_fn=diag_prediction_mimic3_fn,
#                                 dev=args.devm)
# else:
#     task_dataset = load_dataset(args.dataset, root=fileroot[args.dataset])
task_dataset = torch.load(f'task_dataset_{args.dataset}_note.pt')
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
    data_path = './logs/{}_{}_16_note.pkl'.format(args.dataset, args.pe_dim)
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


    # train_loader, val_loader, test_loader = mm_dataloader(trainset, validset, testset, batch_size=args.batch_size)
    model = TRANS(Tokenizers, 128, len(task_dataset.get_all_tokens('conditions')),
                  device, graph_meta=graph_meta, pe=args.pe_dim)

ckptpath = './logs/trained_{}_{}.ckpt'.format(args.model, args.dataset)
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
# best = float('inf')
best = 12345
pbar = tqdm(range(args.epochs))
model.to(device)
# scaler = GradScaler()
    # Run a dummy forward pass to initialize all parameters
    # dummy_input = torch.randn(1, 10).to(torch.device('cuda'))
    # model(dummy_input)

model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank,
                                                      find_unused_parameters=True)
for epoch in pbar:
    # model = model.to(device)
      # 这句加载到多GPU上

    train_loss = train(train_loader, model, label_tokenizer, optimizer, device)
    # train_loss = train(train_loader, model, label_tokenizer, optimizer, device, scaler)
    # val_loss = valid(valid_loader, model, label_tokenizer, device)

    if local_rank == 0:
        pbar.set_description(f"Epoch {epoch + 1}/{args.epochs} - train loss: {train_loss:.2f}")
        # pbar.set_description(
        #     f"Epoch {epoch + 1}/{args.epochs} - train loss: {train_loss:.2f} - valid loss: {val_loss:.2f}")
        # if val_loss < best:
            # print(f"Validation loss improved from {best:.2f} to {val_loss:.2f}. Saving model...")
            # torch.save(model.module.state_dict(), ckptpath)
            # best = val_loss
            # best_model = torch.load(ckptpath)
            # model.load_state_dict(best_model)
            # model = model.to(device)

        y_t_all, y_p_all = [], []
        y_true, y_prob = test(test_loader, model, label_tokenizer)
        print(code_level(y_true, y_prob))
        print(visit_level(y_true, y_prob))
        # del y_true, y_prob
        # torch.cuda.empty_cache()
    # del train_loss
    # torch.cuda.empty_cache()

    # if val_loss < best and local_rank == 1:
    #     torch.save(model.module.state_dict(), './trained_{}_{}_1.ckpt'.format(args.model, args.dataset))
    # break
# for limited gpu memory
# if args.model == 'TRANS':
#     del model
#     torch.cuda.empty_cache()
#     import gc
#
#     gc.collect()
#     device = torch.device('cpu')
#     model = TRANS(Tokenizers, 128, len(task_dataset.get_all_tokens('conditions')),
#                   device, graph_meta=graph_meta, pe=args.pe_dim)
# if local_rank == 0:
#     best_model = torch.load(ckptpath)
#     model.load_state_dict(best_model)
#     model = model.to(device)
#
#     y_t_all, y_p_all = [], []
#     y_true, y_prob = test(test_loader, model, label_tokenizer)
#     print(code_level(y_true, y_prob))
#     print(visit_level(y_true, y_prob))
