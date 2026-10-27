'''
Efficiency comparison on ColorMNIST (100% reversed-colour test set): steering vectors (ActAdd, no training)
vs CD-lite (CAV at conv2 + concept loss fine-tuning, no teacher; ~ "Ours w/o Teacher" in the CD paper).
Both use the same concept set: N digits in their (biased) train colour vs the same digits in grayscale.
Logs accuracy vs wall-clock so the two can be put on one curve.
'''
import os, json, time, argparse
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
from sklearn.linear_model import LogisticRegression
from utils.networks_classi import MNISTColorNet

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--n_pairs_list', type=str, default='1,10,100,1000')
parser.add_argument('--lams', type=str, default='10,100,1000')
parser.add_argument('--epochs', type=int, default=3)
parser.add_argument('--eval_every', type=int, default=100)
parser.add_argument('--out', type=str, default='sv_out/eff_seed0.json')
opt = parser.parse_args()

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(opt.seed); np.random.seed(opt.seed)
sync = (lambda: torch.cuda.synchronize()) if dev == 'cuda' else (lambda: None)

# same colours / normalisation as sv_debias.py
colors = [np.asarray(c) for c in [(1.0, 0.0, 0.0), (0.944662, 0.524812, 0.0), (0.790511, 0.889325, 0.0), (0.277996, 0.833987, 0.0),
          (0.0, 0.778649, 0.173033), (0.0, 0.723312, 0.562576), (0.0, 0.445316, 0.667974), (0.0, 0.068071, 0.612636),
          (0.247688, 0.0, 0.557298), (0.501961, 0.0, 0.501961)]]
dpath = 'dep/data/ColorMNIST'
x_train = np.load(os.path.join(dpath, 'train_x.npy'))
mean, std = x_train.mean(axis=(0, 2, 3)), x_train.std(axis=(0, 2, 3))
norm = lambda x: torch.Tensor((x - mean[None, :, None, None]) / std[None, :, None, None])
train_x, train_y = norm(x_train), torch.Tensor(np.load(os.path.join(dpath, 'train_y.npy'))).long()
del x_train

def load(name):
    return norm(np.load(os.path.join(dpath, name + '_x.npy'))).to(dev), torch.Tensor(np.load(os.path.join(dpath, name + '_y.npy'))).long().to(dev)
val_x, val_y = load('val')
test_x, test_y = load('test')

def base_model():
    m = MNISTColorNet().to(dev)
    m.load_state_dict(torch.load('mnist/ColorMNIST/orig_model_colorMNIST.pt', map_location=dev))
    return m.eval()

mnist = datasets.MNIST(root='data', train=True, download=False)
def concept_pairs(n):
    idx = np.random.choice(len(mnist), n, replace=False)
    digits = mnist.data[idx].numpy().astype(np.float32)[:, None]
    labels = mnist.targets[idx].numpy()
    return norm(digits * np.stack([colors[k] for k in labels])[:, :, None, None]).to(dev), norm(np.repeat(digits, 3, axis=1)).to(dev)

@torch.no_grad()
def get_conv2(model, x):
    cache = {}
    h = model.conv2.register_forward_hook(lambda m, i, o: cache.__setitem__('a', o))
    model(x); h.remove()
    return cache['a']

@torch.no_grad()
def evaluate(model, x, y, hook=None, bs=2000):
    h = model.conv2.register_forward_hook(hook) if hook else None
    correct = sum((model(x[i:i+bs]).argmax(1) == y[i:i+bs]).sum().item() for i in range(0, len(x), bs))
    if h: h.remove()
    return 100. * correct / len(x)

add_hook = lambda d, c: (lambda m, i, o: o + c * d)
coeffs = [-10, -6, -4, -3, -2, -1.5, -1, -0.5, 0.5, 1, 2]
res = {'seed': opt.seed, 'device': dev, 'sv': {}, 'random': {}, 'cd': {}}
model = base_model()
res['base'] = {'val': evaluate(model, val_x, val_y), 'test': evaluate(model, test_x, test_y)}

# ---------------- SV (ActAdd mean diff), cost = build vector + coefficient sweep on val ----------------
for n in [int(k) for k in opt.n_pairs_list.split(',')]:
    sync(); t0 = time.time()
    x_col, x_gray = concept_pairs(n)
    d = (get_conv2(model, x_col) - get_conv2(model, x_gray)).mean(0)
    sync(); t_vec = time.time() - t0
    sweep = {c: evaluate(model, val_x, val_y, add_hook(d, c)) for c in coeffs}
    best_c = max(sweep, key=sweep.get)
    sync(); t_total = time.time() - t0
    res['sv'][n] = {'coeff': best_c, 'val': sweep[best_c], 'test': evaluate(model, test_x, test_y, add_hook(d, best_c)),
                    'sec_vector': t_vec, 'sec_total': t_total}
    # control: random direction with the same norm, same sweep
    rnd = torch.randn(d.shape, device=dev); rnd = rnd / rnd.norm() * d.norm()
    rs = {c: evaluate(model, val_x, val_y, add_hook(rnd, c)) for c in coeffs}
    rc = max(rs, key=rs.get)
    res['random'][n] = {'coeff': rc, 'val': rs[rc], 'test': evaluate(model, test_x, test_y, add_hook(rnd, rc))}
    print('SV n=%d' % n, res['sv'][n], 'random', res['random'][n], flush=True)

# ---------------- CD-lite: CAV (logistic regression) + concept loss |cos(mean_batch grad_conv2 logit_y, cav)| ----------------
n_cav = 1000
for lam in [float(k) for k in opt.lams.split(',')]:
    torch.manual_seed(opt.seed); np.random.seed(opt.seed)
    model = base_model()
    sync(); t0 = time.time()
    x_col, x_gray = concept_pairs(n_cav)
    a = torch.cat([get_conv2(model, x_col), get_conv2(model, x_gray)]).flatten(1).cpu().numpy()
    lr_clf = LogisticRegression(C=0.1, max_iter=1000).fit(a, np.r_[np.ones(n_cav), np.zeros(n_cav)])
    cav = torch.Tensor(lr_clf.coef_[0]).to(dev); cav = cav / cav.norm()
    sync(); t_cav = time.time() - t0

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    curve, best, step, t_train = [], {'val': -1}, 0, 0.
    for ep in range(opt.epochs):
        perm = torch.randperm(len(train_x))
        for i in range(0, len(train_x), 64):
            sync(); ts = time.time()
            model.train()
            b = perm[i:i+64]; x, y = train_x[b].to(dev), train_y[b].to(dev)
            cache = {}
            h = model.conv2.register_forward_hook(lambda m, i_, o: cache.__setitem__('a', o))
            out = model(x); h.remove()
            g = torch.autograd.grad(out.gather(1, y[:, None]).sum(), cache['a'], create_graph=True)[0].flatten(1)
            l_c = F.cosine_similarity(g.mean(0), cav, dim=0).abs()   # batch-level (TCAV-style) sensitivity to colour
            loss = F.nll_loss(out, y) + lam * l_c
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            sync(); t_train += time.time() - ts; step += 1
            if step % opt.eval_every == 0:
                model.eval()
                v, te = evaluate(model, val_x, val_y), evaluate(model, test_x, test_y)
                curve.append({'step': step, 'sec': t_cav + t_train, 'val': v, 'test': te, 'l_c': l_c.item()})
                if v > best['val']: best = dict(curve[-1])
        print('CD lam=%g ep %d' % (lam, ep), curve[-1], flush=True)
    res['cd'][lam] = {'sec_cav': t_cav, 'best_on_val': best, 'curve': curve}
    print('CD lam=%g best' % lam, best, flush=True)

json.dump(res, open(opt.out, 'w'), indent=1)
