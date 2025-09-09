import json
from pathlib import Path
import dgl
import torch
from torch import optim
import torch.nn.functional as F
import numpy as np

import time
import argparse
import os
from tqdm import tqdm, trange

from tfe_models import TFE_GNN, TFE_GNN_large
from tfe_utils import propagate_adj, set_seed, accuracy, load_data, random_walk_adj, consis_loss, matrix_power_sparse

# hardcoded output settings
_RESULT_DIR = Path("results/records")
_RESULT_FORMAT = "jsonl"   # change to "csv" if you prefer CSV aggregation

def train(model, optimizer, adj_hp, adj_lp, x, y, mask):
    model.train()
    optimizer.zero_grad()
    out = model(adj_hp, adj_lp, x)
    out = F.log_softmax(out, dim=1)
    loss = F.cross_entropy(out[mask[0]], y[mask[0]])
    if args.dataset in {'citeseer'} and not args.full:
        cos_loss = consis_loss(out, 0.5, 0.9)
        (loss+cos_loss).backward()
    else:
        loss.backward()
    optimizer.step()
    del out


def test(model, adj_hp, adj_lp, x, y, mask):
    model.eval()
    logits, accs, losses = model(adj_hp, adj_lp, x), [], []
    logits = F.log_softmax(logits, dim=1)
    for i in range(3):
        acc = accuracy(logits[mask[i]], y[mask[i]])
        loss = F.cross_entropy(logits[mask[i]], y[mask[i]])
        accs.append(acc)
        losses.append(loss)

    return accs, losses, logits


def run(args, dataset, optimi, full, random_split, i):
    if args.random_split:
        set_seed(args.seed)
    else:
        set_seed(i)
    device = torch.device('cuda:' + str(args.device) if torch.cuda.is_available() else 'cpu')
    adj, features, labels, train_mask, val_mask, test_mask = load_data(dataset, full, random_split, args.train_rate, args.val_rate, i)
    if args.dataset in {'physics', 'cora-full'}:
        model = TFE_GNN_large(features.shape[1], args.hidden, int(max(labels)) + 1, args.layers, args.dropout,
                                   args.activation, args.hop, args.combine)
    else:
        model = TFE_GNN(features.shape[1], args.hidden, int(max(labels)) + 1, args.layers, args.dropout,
                               args.activation, args.hop, args.combine)
    if optimi == 'Adam':
        optimizer = optim.Adam(
            [
                {'params': model.adaptives, 'weight_decay': args.wd_adaptive, 'lr': args.lr_adaptive},
                # {'params': model.adaptive_lp, 'weight_decay': args.wd_adaptive, 'lr': args.lr_adaptive},
                {'params': model.layers.parameters(), 'weight_decay': args.wd_lin, 'lr': args.lr_lin},
                {'params': model.ense_coe, 'weight_decay': args.wd_adaptive2, 'lr': args.lr_adaptive2},
            ]
        )
    if optimi == "RMSprop":
        optimizer = optim.RMSprop(
            [
                {'params': model.adaptive, 'weight_decay': args.wd_adaptive, 'lr': args.lr_adaptive},
                {'params': model.adaptive_lp, 'weight_decay': args.wd_adaptive, 'lr': args.lr_adaptive},
                {'params': model.layers.parameters(), 'weight_decay': args.wd_lin, 'lr': args.lr_lin},
                {'params': model.ense_coe, 'weight_decay': args.wd_adaptive2, 'lr': args.lr_adaptive2},
            ]
        )


    model.to(device)
    features = features.clone().detach().to(device)
    labels = labels.clone().detach().to(device)
    train_mask = train_mask.clone().detach().to(device)
    val_mask = val_mask.clone().detach().to(device)
    test_mask = test_mask.clone().detach().to(device)
    mask = [train_mask, val_mask, test_mask]

    if args.gf == 'sym':
        adj_lp = propagate_adj(adj, 'low', -0.5, -0.5).to(device)
        adj_hp = propagate_adj(adj, 'high', args.eta, args.eta).to(device)
    elif args.gf == 'rw':
        adj_lp = random_walk_adj(adj, 'low', -1.).to(device)
        adj_hp = random_walk_adj(adj, 'high', -1.).to(device)
    else:
        print("Unsupported Graph Filter Forms")


    #DEBUG PRINTS
    print(f"adj_hp dtyp is {adj_hp.dtype}")        # must be float32/float64/complex
    print(f"adj_hp is sparse {adj_hp.is_sparse}")    # matrix_power does not support sparse tensors
    print(f"adj_hp is contiguous {adj_hp.is_contiguous()}")

    print(f"adj_lp dtyp is {adj_lp.dtype}")        # must be float32/float64/complex
    print(f"adj_lp is sparse {adj_lp.is_sparse}")    # matrix_power does not support sparse tensors
    print(f"adj_lp is contiguous {adj_lp.is_contiguous()}")

    # Build adjacency list for multi-band TFE
    # Default: use LP for all mid bands and HP for the last band
    adjs = []
    if hasattr(model, 'adaptives') and len(model.adaptives) > 2:
        num_bands = len(model.adaptives)
        adjs = [adj_lp]
        
        if args.bandwidths:
            bandwidths = [int(x) for x in args.bandwidths.split(',')]
        else:
            bandwidths = [1] * (num_bands - 2) * 2
        assert len(bandwidths) == (num_bands - 2) * 2, \
            f"incorrect number of bandwidths/bands: bandwidth params {len(bandwidths)}, num bands {num_bands}"

        for b in range(num_bands - 2):     # all LP-like bands
            lower = bandwidths[b*2]
            upper = bandwidths[b*2+1]
            
            hp_part = matrix_power_sparse(adj_hp, lower)
            lp_part = matrix_power_sparse(adj_lp, upper)
            
            print(f"hp_part dtyp is {hp_part.dtype}")        # must be float32/float64/complex
            print(f"hp_part is sparse {hp_part.is_sparse}")    # matrix_power does not support sparse tensors
            print(f"hp_part is contiguous {hp_part.is_contiguous()}")

            print(f"lp_part dtyp is {lp_part.dtype}")        # must be float32/float64/complex
            print(f"lp_part is sparse {lp_part.is_sparse}")    # matrix_power does not support sparse tensors
            print(f"lp_part is contiguous {lp_part.is_contiguous()}")


            print(f"DEBUG: midband {b + 1}, lower {lower}, upper {upper}")
            cur = torch.mm(hp_part, lp_part)  # LP^k1 * HP^k2
            cur = cur.to_sparse()
            adjs.append(cur)
        adjs.append(adj_hp)                # last band is HP
    else:
        adjs = [adj_lp, adj_hp]

    best_acc, best_val_acc, test_acc, best_val_loss = 0, 0, 0, float("inf")
    train_losses = []
    val_losses = []
    run_time = []
    for epoch in trange(args.epochs, desc=f'Run {i+1}'):
        t0 = time.time()
        train(model, optimizer, adjs, None, features, labels, mask)
        run_time.append(time.time()-t0)
        [train_acc, val_acc, tmp_test_acc], [train_loss, val_loss, tmp_test_loss], logits = test(model, adjs, None, features, labels, mask)
        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            test_acc = tmp_test_acc
            bad_epoch = 0

            ada = model.adaptive.data.cpu()
            ada_lp = model.adaptive_lp.data.cpu()
        else:
            bad_epoch += 1
        if bad_epoch == args.patience:
            break
    #torch.save(train_losses, f"/home/user/duan/four_mix_ChebNet_1/eigenvalue/{args.dataset}_tfe_train_loss_mlrrm{i}.pkl")
    #torch.save(val_losses, f"/home/user/duan/four_mix_ChebNet_1/eigenvalue/{args.dataset}_tfe_val_loss_mlrrm{i}.pkl")

    return test_acc, best_val_loss, ada, ada_lp, run_time


parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--dataset', type=str, default='squirrel', help='texas, cornell, wisconsin, chameleon, squirrel, cora'
                                                                    'citeseer, pubmed, cora-full, cs, physics')
parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
parser.add_argument('--patience', type=int, default=200, help='Patience')
parser.add_argument('--hidden', type=int, default=512, help='Number of hidden units.')
parser.add_argument('--layers', type=int, default=2, help='')
parser.add_argument('--device', type=int, default=0, help='GPU device.')
parser.add_argument('--runs', type=int, default=10, help='number of runs.')

parser.add_argument('--optimizer', type=str, default='Adam', help="Adam, RMSprop")
parser.add_argument('--hop_lp', type=int, default=6, help='K_lp in our paper')
parser.add_argument('--hop_hp', type=int, default=5, help='K_hp in our paper')
parser.add_argument('--pro_dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability) of propagation.')
parser.add_argument('--lin_dropout', type=float, default=0., help='Dropout rate (1 - keep probability) of linear.')
parser.add_argument('--eta', type=float, default=-0.3, help='exponent of H_hp')

parser.add_argument('--bands', type=int, default=2, help='Number of filter bands (>=2).')
parser.add_argument('--hops', type=str, default=None, help='Comma-separated per-band K values, e.g., "6,4,2,5".')

parser.add_argument('--lr_adaptive', type=float, default=0.1, help='Initial learning rate of coefficients.')
parser.add_argument('--wd_adaptive', type=float, default=0.05, help='Weight decay (L2 loss on parameters) of coefficients.')
parser.add_argument('--lr_adaptive2', type=float, default=0.00, help='Initial learning rate of coefficients.')
parser.add_argument('--wd_adaptive2', type=float, default=0.000, help='Weight decay (L2 loss on parameters) of coefficients.')
parser.add_argument('--lr_lin', type=float, default=0.005, help='Initial learning rate of linear.')
parser.add_argument('--wd_lin', type=float, default=0.0, help='Weight decay (L2 loss on parameters) of linear.')

parser.add_argument('--gf', type=str, default='sym', help="H_hp, H_lp: sym, rw")
parser.add_argument('--activation', type=bool, default=True)
parser.add_argument('--full', type=bool, default=True, help='Whether full-supervised')
parser.add_argument('--random_split', type=bool, default=True, help='Whether random split')
parser.add_argument('--combine', type=str, default='sum', help='sum, con, lp, hp')

parser.add_argument('--bandwidths', type=str, default=None, help='Comma-separated per-band bandwidths given as tuples, exp for HP and exp for LP, e.g., "1,3,2,4".')



args = parser.parse_args()
print(args)

args.dropout = [args.pro_dropout, args.lin_dropout]
args.hop = [int(x) for x in args.hops.split(',')] if args.hops else ([args.hop_lp]*(args.bands-1) + [args.hop_hp])

if args.full:
    args.train_rate = 0.6
    args.val_rate = 0.2
else:
    args.train_rate = 0.025
    args.val_rate = 0.025

results = []
time_results=[]
all_test_accs = []
adas = []
adas_lp = []

for i in range(args.runs):
    test_acc, best_val_loss, ada, ada_lp, run_time = run(args, args.dataset, args.optimizer, args.full, args.random_split, i)
    time_results.append(run_time)
    results.append([ada, ada_lp])
    all_test_accs.append(test_acc.item())
    print(f'run_{str(i+1)} \t test_acc: {test_acc:.4f}')
run_sum=0
epochsss=0
for i in time_results:
    run_sum+=sum(i)
    epochsss+=len(i)
#print(results)
print("each run avg_time:",run_sum/(args.runs),"s")
print("each epoch avg_time:",1000*run_sum/epochsss,"ms")
print('test acc mean (%) =', np.mean(all_test_accs)*100, np.std(all_test_accs)*100)


# --- append one-line result record (hardcoded) ---

# ensure directory exists
_RESULT_DIR.mkdir(parents=True, exist_ok=True)

# collect canonical strings (avoid breaking if attr missing)
_dataset   = getattr(args, "dataset", "unknown")
_bands     = int(getattr(args, "bands", 0))
_seed      = int(getattr(args, "seed", 0))
_runs      = int(getattr(args, "runs", 1))
_bandwidths = getattr(args, "bandwidths", "")  # recorded verbatim if present
# hops could be in args.hops (csv string) or args.hop (list); normalize:
if hasattr(args, "hops") and isinstance(args.hops, str) and args.hops:
    _hops_str = args.hops
elif hasattr(args, "hop") and isinstance(args.hop, (list, tuple)):
    _hops_str = ",".join(str(x) for x in args.hop)
else:
    _hops_str = ""

# final accuracies: expect all_test_accs to be fractions in [0,1]
# if yours are percents, divide by 100.0 here.
_acc_mean = float(np.mean(all_test_accs))
_acc_std  = float(np.std(all_test_accs))

# add slurm job id if running under slurm
_job_id = os.environ.get("SLURM_JOB_ID")

_record = {
    "dataset":    _dataset,
    "bands":      _bands,
    "bandwidths": _bandwidths,   # e.g. "1,3,5,10" or "1,3,1,3" etc.
    "hops":       _hops_str,     # e.g. "5,5,5"
    "seed":       _seed,
    "runs":       _runs,
    "acc_mean":   _acc_mean,     # fraction (e.g., 0.7423)
    "acc_std":    _acc_std,      # fraction
    "job_id":     _job_id,
}

_out_path = _RESULT_DIR / f"{_dataset}__bands{_bands}.{_RESULT_FORMAT}"

if _RESULT_FORMAT == "jsonl":
    with _out_path.open("a", encoding="utf-8") as _f:
        _f.write(json.dumps(_record, ensure_ascii=False) + "\n")
else:  # CSV fallback
    _header = ["dataset","bands","bandwidths","hops","seed","runs","acc_mean","acc_std","job_id"]
    _exists = _out_path.exists()
    with _out_path.open("a", encoding="utf-8") as _f:
        if not _exists:
            _f.write(",".join(_header) + "\n")
        _f.write(",".join([
            str(_record["dataset"]),
            str(_record["bands"]),
            str(_record["bandwidths"]),
            str(_record["hops"]),
            str(_record["seed"]),
            str(_record["runs"]),
            f"{_record['acc_mean']:.6f}",
            f"{_record['acc_std']:.6f}",
            str(_record["job_id"] or ""),
        ]) + "\n")
# --- end: result record ---
