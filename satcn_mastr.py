"""Train one SATCN per dataset on our splits and score it on all four structures.

    python tools_export_satcn.py --dataset metr_la          # in the MaSTR repo
    python satcn_mastr.py --npz ../mastr-tmc/satcn_data/metr_la_seed1.npz \
        --device cuda:0 --epochs 20

SATCN is a kriging model, but its masking is element-wise rather than per node:
the per-step adjacency clears the column of every entry whose input is zero
(datasets.py:175-176), so a hole at an observed node and an uncovered node are
handled by the same mechanism. That is what lets one trained model be scored on
Point, Frame and Block as well as Sensor, and it is worth stating in its own
right, since the other kriging baselines have no such path.

Four things here depart from the released demo, each because the demo cannot be
used for a comparison as it stands.

  The per-step adjacency is built during the forward pass, not precomputed. The
  released pipeline materialises an array of shape (windows, T, N, N) and writes
  it to disk; at our window length that is terabytes on METR-LA and worse on
  USHCN. It is the same tensor either way: the static top-k graph with the
  columns of unread nodes cleared, re-ranked per step.

  Epochs are chosen on a validation split. The demo selects on the test set
  (demo_metr.py:117-119) and reports that same test figure, so its number is not
  a held-out one. Validation here uses the Point mask, the structure SATCN is
  built for and therefore the most favourable selection available to it.

  The loss and the metrics take an explicit mask rather than treating zero as
  missing. On USHCN precipitation zero is the most common real reading, and
  null_val=0 would drop every dry month from both.

  Windows tile the series instead of overlapping, which is the released
  protocol: each position is scored exactly once. Our own stride-1 figures
  average a position once per window containing it, so the MaSTR column to put
  beside this one is the stride-24 evaluation, not the stride-1 one.

The scored positions are our evaluation masks, asserted position by position.
"""
import argparse
import copy
import os
import time

import numpy as np
import torch
import torch.optim as optim

from model import general_satcn
from sample_gene import sadj_transform, ZerooneScaler

STRUCTURES = ['random', 'timepoint', 'region', 'sensor']
PAPER = {'random': 'Point', 'timepoint': 'Frame', 'region': 'Block',
         'sensor': 'Sensor'}
EPS = 1e-8


# ── windows ──────────────────────────────────────────────────────────────────

def train_starts(T, score_len, stride):
    """Window starts for training, which need not tile.

    The evaluation has to tile: each position is scored exactly once, which is
    the released protocol and what makes the position count comparable with
    ours. Training carries no such constraint, and reusing the tiling there
    throws away almost all of the data. On BJ-AIR a stride equal to the scored
    length leaves 123 windows, which at batch 8 over 20 epochs is 320 gradient
    steps; SATCN's own demo gets about ten thousand, because its scored length
    is 4 rather than 24. A stride of one restores that, and a larger stride is
    there for the datasets where it costs too much time.
    """
    last = max(T - score_len, 0)
    return list(range(0, last + 1, stride))


def tile(T, score_len, t_reduce):
    """Starts of the scored segments, covering every step exactly once.

    The series is treated as if it carried t_reduce empty steps in front, so the
    first segment can be scored like any other: those steps are masked out and
    contribute nothing. The tail is padded for the same reason. Coverage is then
    exact, which is what makes the position count comparable with ours.
    """
    return list(range(0, T, score_len))


def gather(data, mask, starts, score_len, t_reduce):
    """-> x (B, 1, score_len + t_reduce, N) and the mask of the same shape.

    Index -1 and beyond are the pad, so they are read as absent rather than
    wrapped around, which indexing with a negative number would do silently.
    """
    T, N = data.shape
    L = score_len + t_reduce
    xb = np.zeros((len(starts), 1, L, N), dtype=np.float32)
    mb = np.zeros((len(starts), 1, L, N), dtype=np.float32)
    for i, s in enumerate(starts):
        lo, hi = s - t_reduce, s - t_reduce + L
        src_lo, src_hi = max(lo, 0), min(hi, T)
        if src_hi <= src_lo:
            continue
        dst_lo = src_lo - lo
        xb[i, 0, dst_lo:dst_lo + (src_hi - src_lo)] = data[src_lo:src_hi]
        mb[i, 0, dst_lo:dst_lo + (src_hi - src_lo)] = mask[src_lo:src_hi]
    return xb, mb


def scored_slice(arr, starts, score_len):
    """The target laid out to match the model's output, (B, 1, score_len, N)."""
    T, N = arr.shape
    out = np.zeros((len(starts), 1, score_len, N), dtype=np.float32)
    for i, s in enumerate(starts):
        hi = min(s + score_len, T)
        if hi > s:
            out[i, 0, :hi - s] = arr[s:hi]
    return out


# ── graph ────────────────────────────────────────────────────────────────────

def step_adj(A2, mask_bt, least_k):
    """The masked, re-ranked per-step adjacency, (B, T, N, N).

    A node that carries no reading at a step sends no message at that step, so
    its column is cleared; the top-k is then taken per row, exactly as
    sadj_transform does, including its treatment of ties and of a row left with
    fewer than k edges.
    """
    A = A2.unsqueeze(0).unsqueeze(0) * mask_bt.unsqueeze(-2)
    kth = A.topk(least_k, dim=-1).values[..., -1:]
    return torch.where(A >= kth, A, torch.zeros((), device=A.device))


def isolated_fraction(A):
    """Share of (sample, step, node) with no surviving neighbour.

    aggregate_mean divides by the neighbour count, so these positions come out
    NaN and are replaced by the constant 6 in model.py. On Frame every node at a
    held-out step is isolated, and the figure for that structure is produced
    almost entirely by that fallback and by the temporal convolution that
    follows. Reporting the share keeps that from being read as a model result.
    """
    return float((A.sum(dim=-1) <= 0).float().mean())


# ── metrics ──────────────────────────────────────────────────────────────────

def masked_mae_t(pred, truth, mask):
    d = (pred - truth).abs() * mask
    return d.sum() / mask.sum().clamp(min=1.)


def metrics(pred, truth, scored):
    e = pred[scored] - truth[scored]
    y = truth[scored]
    den = np.abs(y)
    ok = den > EPS
    return {'mae': float(np.abs(e).mean()),
            'rmse': float(np.sqrt((e ** 2).mean())),
            'mape': float((np.abs(e[ok]) / den[ok]).mean()),
            'mre': float(np.abs(e).sum() / (den.sum() + EPS))}


# ── running the model ────────────────────────────────────────────────────────

def run_split(model, scaler, data, raw_mask, eval_mask, A2, args, device,
              report=None):
    """Predict the scored segments of one split and return them in the original unit.

    The input is raw_mask * (1 - eval_mask), which is how our own pipeline forms
    it, so the model is given exactly what our models are given.
    """
    t_reduce = (args.layers + 1) * (args.t_kernel - 1)
    starts = tile(data.shape[0], args.score_len, t_reduce)
    in_mask = raw_mask * (1. - eval_mask)
    xb, mb = gather(data * in_mask, in_mask, starts, args.score_len, t_reduce)

    preds = np.zeros((len(starts), 1, args.score_len, data.shape[1]),
                     dtype=np.float32)
    iso = []
    model.eval()
    with torch.no_grad():
        for lo in range(0, len(starts), args.test_batch_size):
            hi = min(lo + args.test_batch_size, len(starts))
            x = torch.from_numpy(scaler.transform(xb[lo:hi])).to(device)
            m = torch.from_numpy(mb[lo:hi, 0]).to(device)
            A1 = step_adj(A2, m, args.least_k)
            iso.append(isolated_fraction(A1))
            out = model(x, A2, A1)
            preds[lo:hi] = scaler.inverse_transform(out).cpu().numpy()
    if report is not None:
        report['isolated_fraction'] = round(float(np.mean(iso)), 4)
    return preds, starts


def score(preds, starts, data, eval_mask, args, name, report):
    """Score on our evaluation mask and refuse if the position sets differ."""
    truth = scored_slice(data, starts, args.score_len)
    sc = scored_slice(eval_mask, starts, args.score_len) > 0.5
    want = int(eval_mask.sum())
    got = int(sc.sum())
    if got != want:
        raise SystemExit(
            'the tiling covers %d of the %d positions our eval_mask marks for '
            '%s, so this run would not be scored on our positions'
            % (got, want, name))
    res = metrics(preds.astype(np.float64), truth.astype(np.float64), sc)
    report.update(res)
    return res, got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--out', default='logs_satcn')
    # geometry, matched to ours
    ap.add_argument('--score-len', type=int, default=24,
                    help='scored segment length; 24 to match our window')
    # SATCN's own defaults, from demo_metr.py
    ap.add_argument('--layers', type=int, default=1)
    ap.add_argument('--t-kernel', type=int, default=2)
    ap.add_argument('--least-k', type=int, default=2)
    ap.add_argument('--nv-frac', type=float, default=20. / 103.,
                    help='share of nodes withheld per step in training. SATCN '
                         'sets this as a count, N_v=20, against the 103 '
                         'training nodes of its METR-LA split, so the share it '
                         'uses is about 0.19; a count carried over to a '
                         'smaller node set withholds everything and the model '
                         'sees an empty input')
    ap.add_argument('--train-stride', type=int, default=1,
                    help='stride of the training windows; the evaluation '
                         'always tiles')
    ap.add_argument('--nv', type=int, default=0,
                    help='absolute node count, overriding --nv-frac')
    ap.add_argument('--channels', type=int, default=128)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--test-batch-size', type=int, default=4,
                    help='the per-step adjacency is (B, T, N, N); use 1 on USHCN')
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--weight-decay', type=float, default=0.0001)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--released-frozen-spatial', action='store_true',
                    help='leave the spatial aggregation weights untrained, as '
                         'the released code does; see model.py')
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available()
                          or 'cpu' in args.device else 'cpu')
    t_reduce = (args.layers + 1) * (args.t_kernel - 1)

    d = np.load(args.npz)
    base = os.path.splitext(os.path.basename(args.npz))[0]
    ds_name = base.split('_seed')[0]
    # The file name may carry a variant after the seed, such as _denseadj.
    tail = base.split('_seed')[1]
    seed = int(''.join(c for c in tail.split('_')[0] if c.isdigit()))
    variant = '_'.join(tail.split('_')[1:])
    adj_source = str(d['adj_source']) if 'adj_source' in d.files else 'unknown'
    scaler = ZerooneScaler(float(d['train_max']))

    print('SATCN on %s  (seed %d, device %s)' % (ds_name, seed, device))
    print('  |V_o|=%d  |V_u|=%d  N=%d' % (int(d['n_known']),
                                          len(d['unknown_set']),
                                          int(d['n_features'])))
    print('  input length %d = %d scored + %d consumed by the convolutions'
          % (args.score_len + t_reduce, args.score_len, t_reduce))
    print('  graph: %s' % ('SATCN\'s own dense distance kernel'
                               if adj_source == 'dense'
                               else 'the adjacency every method is given'))
    print('  spatial aggregation weights: %s'
          % ('frozen, as released' if args.released_frozen_spatial
             else 'trained'))

    # Graphs. Training and the three imputation structures use the induced
    # subgraph on V_o; Sensor uses the full graph, which SATCN can take because
    # it holds no per-node parameter.
    A2_o = torch.from_numpy(
        sadj_transform(np.array(d['adj_known'], dtype=np.float32),
                       args.least_k)).to(device)
    A2_f = torch.from_numpy(
        sadj_transform(np.array(d['adj_full'], dtype=np.float32),
                       args.least_k)).to(device)
    # avg_d is a training-set statistic and stays the training graph's, which is
    # what makes the transfer to the full graph inductive.
    avg_d = {'log': torch.log(A2_o.sum(dim=0) + 1).mean().to(device)}

    model = general_satcn(avg_d, device, layers=args.layers,
                          t_kernel=args.t_kernel, channels=args.channels,
                          train_spatial=not args.released_frozen_spatial
                          ).to(device)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('  trainable parameters: %d' % n_par)
    opt = optim.Adam(model.parameters(), lr=args.lr,
                     weight_decay=args.weight_decay)

    # Training windows, on V_o only.
    tr, m_tr = d['train_data'], d['train_mask']
    starts_tr = train_starts(tr.shape[0], args.score_len, args.train_stride)
    x_tr, mk_tr = gather(tr, m_tr, starts_tr, args.score_len, t_reduce)
    y_tr = scored_slice(tr, starts_tr, args.score_len)
    ym_tr = scored_slice(m_tr, starts_tr, args.score_len)

    # How many nodes to withhold per step. A count large enough to cover the
    # node set empties the input at every step, and the model can then only
    # learn a constant: the loss settles at the value a constant gives, the
    # frozen arm matches the trained one, and least_k stops mattering. That
    # failure produces entirely plausible figures, so it is refused here rather
    # than reported.
    n_o = int(d['n_known'])
    nv = args.nv if args.nv > 0 else int(round(args.nv_frac * n_o))
    nv = max(0, min(nv, n_o - 1))
    if args.nv > 0 and args.nv >= n_o:
        raise SystemExit(
            '--nv %d withholds every one of the %d training nodes at every '
            'step, leaving an empty input; pass --nv-frac instead, or an --nv '
            'below %d' % (args.nv, n_o, n_o))
    print('  train windows: %d (stride %d)   val steps: %d   test steps: %d'
          % (len(starts_tr), args.train_stride, len(d['val_data']),
             len(d['test_data_random'])))
    print('  withholding %d of %d nodes per step (%.0f%%), %d gradient steps '
          'over %d epochs'
          % (nv, n_o, 100. * nv / max(n_o, 1),
             args.epochs * -(-len(starts_tr) // args.batch_size), args.epochs))

    rng = np.random.RandomState(args.seed)
    best, best_val, hist = None, float('inf'), []
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(len(starts_tr))
        losses = []
        for lo in range(0, len(order), args.batch_size):
            idx = order[lo:lo + args.batch_size]
            x = x_tr[idx].copy()
            mk = mk_tr[idx].copy()
            # SATCN's inductive masking: a fresh set of nodes is withheld at
            # every step, which is the training signal its paper describes.
            if nv > 0:
                B, _, L, N = x.shape
                for b in range(B):
                    for t in range(L):
                        hide = rng.choice(N, nv, replace=False)
                        x[b, 0, t, hide] = 0.
                        mk[b, 0, t, hide] = 0.
            xt = torch.from_numpy(scaler.transform(x)).to(device)
            mt = torch.from_numpy(mk[:, 0]).to(device)
            yt = torch.from_numpy(scaler.transform(y_tr[idx])).to(device)
            wt = torch.from_numpy(ym_tr[idx]).to(device)
            A1 = step_adj(A2_o, mt, args.least_k)

            opt.zero_grad()
            pre = model(xt, A2_o, A1)
            loss = masked_mae_t(pre, yt, wt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
            opt.step()
            losses.append(loss.item())

        rep = {}
        pv, sv = run_split(model, scaler, d['val_data'], d['val_mask'],
                           d['val_eval_mask'], A2_o, args, device, rep)
        truth = scored_slice(d['val_data'], sv, args.score_len)
        sc = scored_slice(d['val_eval_mask'], sv, args.score_len) > 0.5
        val_mae = float(np.abs(pv[sc] - truth[sc]).mean())
        hist.append(val_mae)
        print('  epoch %3d  train %.4f   val MAE %.4f%s'
              % (ep, float(np.mean(losses)), val_mae,
                 '   (best)' if val_mae < best_val else ''))
        if val_mae < best_val:
            best_val, best = val_mae, copy.deepcopy(model.state_dict())
    fit_s = time.time() - t0
    if best is not None:
        model.load_state_dict(best)
    print('\nfit took %.1f s, best validation MAE %.4f at epoch %d'
          % (fit_s, best_val, int(np.argmin(hist)) + 1))

    results, extra = {}, {}
    for p in STRUCTURES:
        A2 = A2_f if p == 'sensor' else A2_o
        rep = {}
        preds, starts = run_split(model, scaler, d['test_data_%s' % p],
                                  d['test_mask_%s' % p],
                                  d['test_eval_%s' % p], A2, args, device, rep)
        res, n = score(preds, starts, d['test_data_%s' % p],
                       d['test_eval_%s' % p], args, p, rep)
        results[p] = {k: res[k] for k in ('mae', 'rmse', 'mape', 'mre')}
        extra[p] = {'positions': n,
                    'isolated_fraction': rep.get('isolated_fraction')}
        print('  %-10s (%-6s) MAE=%.4f  RMSE=%.4f  MAPE=%.4f  MRE=%.4f   '
              'n=%d   isolated inputs %.1f%%'
              % (p, PAPER[p], res['mae'], res['rmse'], res['mape'], res['mre'],
                 n, 100. * (rep.get('isolated_fraction') or 0.)))

    # The variant and least_k go in the path: two arms launched in the same
    # minute would otherwise land in the same directory and the first would be
    # overwritten without a word.
    run = os.path.join(args.out,
                       'satcn%s_%s' % ('_' + variant if variant else '', ds_name),
                       '%s_s%dm%d' % (ds_name, seed, seed),
                       '%s_k%d%s' % (time.strftime('%m%d_%H%M'), args.least_k,
                                     '_frozen' if args.released_frozen_spatial
                                     else ''))
    os.makedirs(run, exist_ok=True)
    doc = {
        'config': {
            'data': {'dataset': ds_name, 'seed': seed, 'model_seed': seed},
            'model': {'name': 'SATCN', 'source': 'Kaimaoge/SATCN',
                      'layers': args.layers, 't_kernel': args.t_kernel,
                      'least_k': args.least_k, 'channels': args.channels,
                      'N_v': nv, 'N_v_fraction': round(nv / max(n_o, 1), 4),
                      'train_stride': args.train_stride,
                      'train_spatial': not args.released_frozen_spatial,
                      'adj_source': adj_source, 'variant': variant,
                      'trainable_parameters': n_par},
            'training': {'epochs': args.epochs, 'batch_size': args.batch_size,
                         'lr': args.lr, 'training_type': 'baseline',
                         'selection': 'validation, Point mask'},
            'logging': {'exp_name': 'base_satcn%s_%s'
                                    % ('_' + variant if variant else '',
                                       ds_name)},
        },
        'balanced_checkpoint': {'path': 'best validation epoch',
                                'results': results},
        'evaluation': {'scored_length': args.score_len,
                       'input_length': args.score_len + t_reduce,
                       'stride': args.score_len,
                       'per_structure': extra},
        'timing': {'fit_seconds': round(fit_s, 1)},
        'note': ('Windows tile the series, so each position is scored once; the '
                 'comparable MaSTR column is the stride-%d evaluation. '
                 'isolated_fraction is the share of inputs whose every '
                 'neighbour was absent, where the masked aggregation falls back '
                 'to a constant.' % args.score_len),
    }
    import yaml
    with open(os.path.join(run, 'results.yaml'), 'w', encoding='utf-8') as fh:
        yaml.safe_dump(doc, fh, sort_keys=False, allow_unicode=True)
    print('\nwrote %s/results.yaml' % run)


if __name__ == '__main__':
    main()
