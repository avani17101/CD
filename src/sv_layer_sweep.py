'''
ActAdd layer x coefficient x n_pairs sweep on ColorMNIST (reversed-colour test set), with a same-norm random-vector
control for every cell. Reports the val-selected setting and the best-on-test oracle (an upper bound on what any
hyperparameter choice could reach).
'''
import os, json, argparse
import numpy as np
import torch
import torchvision.datasets as datasets
from utils.networks_classi import MNISTColorNet

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--layers', type=str, default='conv1,conv2,fc1')
parser.add_argument('--n_pairs_list', type=str, default='1,10,100,1000')
parser.add_argument('--out', type=str, default='sv_out/sweep_seed0.json')
opt = parser.parse_args()

torch.set_grad_enabled(False)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(opt.seed); np.random.seed(opt.seed)

# same colours / normalisation as sv_debias.py
colors = [np.asarray(c) for c in [(1.0, 0.0, 0.0), (0.944662, 0.524812, 0.0), (0.790511, 0.889325, 0.0), (0.277996, 0.833987, 0.0),
          (0.0, 0.778649, 0.173033), (0.0, 0.723312, 0.562576), (0.0, 0.445316, 0.667974), (0.0, 0.068071, 0.612636),
          (0.247688, 0.0, 0.557298), (0.501961, 0.0, 0.501961)]]
dpath = 'dep/data/ColorMNIST'
x_train = np.load(os.path.join(dpath, 'train_x.npy'))
mean, std = x_train.mean(axis=(0, 2, 3)), x_train.std(axis=(0, 2, 3))
del x_train
norm = lambda x: torch.Tensor((x - mean[None, :, None, None]) / std[None, :, None, None])

def load(name):
    return norm(np.load(os.path.join(dpath, name + '_x.npy'))).to(dev), torch.Tensor(np.load(os.path.join(dpath, name + '_y.npy'))).long().to(dev)
val_x, val_y = load('val')
test_x, test_y = load('test')

model = MNISTColorNet().to(dev)
model.load_state_dict(torch.load('mnist/ColorMNIST/orig_model_colorMNIST.pt', map_location=dev))
model.eval()

mnist = datasets.MNIST(root='data', train=True, download=False)
n_max = max(int(k) for k in opt.n_pairs_list.split(','))
idx = np.random.choice(len(mnist), n_max, replace=False)
digits = mnist.data[idx].numpy().astype(np.float32)[:, None]
labels = mnist.targets[idx].numpy()
x_col = norm(digits * np.stack([colors[k] for k in labels])[:, :, None, None]).to(dev)
x_gray = norm(np.repeat(digits, 3, axis=1)).to(dev)

def get_act(layer, x):
    cache = {}
    h = getattr(model, layer).register_forward_hook(lambda m, i, o: cache.__setitem__('a', o))
    model(x); h.remove()
    return cache['a']

def evaluate(x, y, layer=None, d=None, c=0., bs=2000):
    h = getattr(model, layer).register_forward_hook(lambda m, i, o: o + c * d) if layer else None
    correct = sum((model(x[i:i+bs]).argmax(1) == y[i:i+bs]).sum().item() for i in range(0, len(x), bs))
    if h: h.remove()
    return 100. * correct / len(x)

mags = [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1, 1.5, 2, 3, 5, 7.5, 10, 20, 50]
coeffs = sorted([-m for m in mags] + mags)
res = {'seed': opt.seed, 'coeffs': coeffs, 'base_test': evaluate(test_x, test_y), 'cells': []}
g = torch.Generator(device='cpu').manual_seed(opt.seed)
for layer in opt.layers.split(','):
    diff = get_act(layer, x_col) - get_act(layer, x_gray)
    for n in [int(k) for k in opt.n_pairs_list.split(',')]:
        d = diff[:n].mean(0)
        rnd = torch.randn(d.shape, generator=g).to(dev); rnd = rnd / rnd.norm() * d.norm()
        for kind, vec in [('sv', d), ('random', rnd)]:
            val = [evaluate(val_x, val_y, layer, vec, c) for c in coeffs]
            test = [evaluate(test_x, test_y, layer, vec, c) for c in coeffs]
            iv, it = int(np.argmax(val)), int(np.argmax(test))
            cell = {'layer': layer, 'n_pairs': n, 'kind': kind, 'val': val, 'test': test,
                    'val_selected': {'coeff': coeffs[iv], 'test': test[iv]}, 'oracle': {'coeff': coeffs[it], 'test': test[it]}}
            res['cells'].append(cell)
            print(layer, n, kind, 'val-selected', cell['val_selected'], 'oracle', cell['oracle'], flush=True)
json.dump(res, open(opt.out, 'w'), indent=1)
