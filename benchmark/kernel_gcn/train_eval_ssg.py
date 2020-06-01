import time
import pdb
import torch
import torch.nn.functional as F
from torch import tensor
from torch.optim import Adam
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import DataLoader, DenseDataLoader as DenseLoader
from datasets import MyDenseCollater, MyDenseDataLoader
import torch_geometric.transforms as T

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def pre_eig(data):
    adj= data.adj
    adj = ((adj + adj.transpose(1, 2)) > 0.).float()
    diag_ele = torch.sum(adj, -1)
    Diag = torch.diag_embed(diag_ele)
    Lap1 = Diag - adj
    e, v = torch.symeig(Lap1.cpu(), eigenvectors=True)

    return e,v
def cross_validation_with_val_set(dataset, model, folds, epochs, batch_size,
                                  lr, lr_decay_factor, lr_decay_step_size,
                                  weight_decay, logger=None, diff=False):
    
    val_losses, accs, durations = [], [], []
    global_iter =1
    for fold, (train_idx, test_idx, val_idx) in enumerate(zip(*k_fold(dataset, folds))):
        for i in range(global_iter):
            train_dataset = dataset[train_idx[i]]
            test_dataset = dataset[test_idx[i]]
            val_dataset = dataset[val_idx[i]]

            if 'adj' in train_dataset[0]:
                #train_loader = MyDenseDataLoader(train_dataset, batch_size, shuffle=True, collate_fn=MyDenseCollater())
                train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
                test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
            else:
                train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
                test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

            model.to(device).reset_parameters()
            optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            t_start = time.perf_counter()

            for epoch in range(1, epochs + 1):
                if(diff):
                    train_loss = train_diff(model, optimizer, train_loader)
                    val_losses.append(eval_loss_diff(model, val_loader))
                    accs.append(eval_acc_diff(model, test_loader))
                else: 
                    train_loss = train(model, optimizer, train_loader)
                    val_losses.append(eval_loss(model, val_loader))
                    accs.append(eval_acc(model, test_loader))
                eval_info = {
                    'fold': fold,
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_losses[-1],
                    'test_acc': accs[-1],
                }

                if logger is not None:
                    logger(eval_info)

                if epoch % lr_decay_step_size == 0:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_decay_factor * param_group['lr']

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            t_end = time.perf_counter()
            durations.append(t_end - t_start)

    loss, acc, duration = tensor(val_losses), tensor(accs), tensor(durations)
    loss, acc = loss.view(global_iter,folds, epochs), acc.view(global_iter,folds, epochs)
    loss = loss.mean(1)
    acc = acc.mean(1)

    loss, argmin = loss.min(dim=1)
    # pdb.set_trace
    acc = acc[torch.arange(folds, dtype=torch.long), argmin]

    loss_mean = loss.mean().item()
    acc_mean = acc.mean().item()
    acc_std = acc.std().item()
    duration_mean = duration.mean().item()
    print('Val Loss: {:.4f}, Test Accuracy: {:.3f} ± {:.3f}, Duration: {:.3f}'.
          format(loss_mean, acc_mean, acc_std, duration_mean))

    return loss_mean, acc_mean, acc_std


def k_fold(dataset, folds):
    train_all=[]
    test_all=[]
    val_all=[]
    for i in range(10):
        skf = StratifiedKFold(folds, shuffle=True, random_state=i)

        test_indices, train_indices = [], []
        for _, idx in skf.split(torch.zeros(len(dataset)), dataset.data.y):
            test_indices.append(torch.from_numpy(idx))

        val_indices = [test_indices[i - 1] for i in range(folds)]

        for i in range(folds):
            train_mask = torch.ones(len(dataset), dtype=torch.bool)
            train_mask[test_indices[i]] = 0
            train_mask[val_indices[i]] = 0
            train_indices.append(train_mask.nonzero().view(-1))
        train_all.append(train_indices)
        test_all.append(test_indices)
        val_all.append(val_indices)
    return train_all, test_all, val_all


def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)


def train(model, optimizer, loader):
    model.train()

    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        data = data.to(device)
        out = model(data)
        loss = F.nll_loss(out, data.y.view(-1))
        loss.backward()
        total_loss += loss.item() * num_graphs(data)
        optimizer.step()
    return total_loss / len(loader.dataset)

def train_diff(model, optimizer, loader):
    model.train()

    total_loss = 0
    for data in loader:
        # pdb.set_trace()
        optimizer.zero_grad()
        data = data.to(device)
        out, extra_loss = model(data)
        loss = F.nll_loss(out, data.y.view(-1))
        loss += 1.0 * extra_loss
        loss.backward()
        total_loss += loss.item()  * num_graphs(data)
        optimizer.step()
    return total_loss / len(loader.dataset)

def eval_acc(model, loader):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data).max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
    return correct / len(loader.dataset)

def eval_acc_diff(model, loader):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)

        with torch.no_grad():
            out, _ =model(data)
            pred = out.max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
    return correct / len(loader.dataset)


def eval_loss(model, loader):
    model.eval()

    loss = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            out = model(data)
        loss += F.nll_loss(out, data.y.view(-1), reduction='sum').item()
    return loss / len(loader.dataset)

def eval_loss_diff(model, loader):
    model.eval()

    loss = 0
    for data in loader:

        data = data.to(device)

        with torch.no_grad():
            out, _ = model(data)
        loss += F.nll_loss(out, data.y.view(-1), reduction='sum').item()
    return loss / len(loader.dataset)
