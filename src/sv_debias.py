'''
ActAdd / steering vectors (Turner et al., sv.ipynb) applied to CD's ColorMNIST debiasing task.
No training: take conv2 activations of a contrast pair (coloured digit vs the same digit in grayscale),
act_diff = act(coloured) - act(gray), and add coeff*act_diff to conv2 at test time (coeff<0 => remove colour).
'''
import os, json, argparse
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
from utils.networks_classi import MNISTColorNet

parser = argparse.ArgumentParser()
parser.add_argument('--n_pairs', type=int, default=1000)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--out', type=str, default='sv_results.json')
opt = parser.parse_args()

torch.set_grad_enabled(False)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(opt.seed); np.random.seed(opt.seed)

# same colours / normalisation as mnist/ColorMNIST/00_make_data.py and utils_train.get_model_and_data
# = [x.get_rgb() for x in Color("red").range_to(Color("purple"), 10)] (colour pkg), hardcoded
colors = [np.asarray(c) for c in [(1.0, 0.0, 0.0), (0.944662, 0.524812, 0.0), (0.790511, 0.889325, 0.0), (0.277996, 0.833987, 0.0),
          (0.0, 0.778649, 0.173033), (0.0, 0.723312, 0.562576), (0.0, 0.445316, 0.667974), (0.0, 0.068071, 0.612636),
          (0.247688, 0.0, 0.557298), (0.501961, 0.0, 0.501961)]]
dpath = 'dep/data/ColorMNIST'
x_train = np.load(os.path.join(dpath, 'train_x.npy'))
mean, std = x_train.mean(axis=(0, 2, 3)), x_train.std(axis=(0, 2, 3))
del x_train
norm = lambda x: torch.Tensor((x - mean[None, :, None, None]) / std[None, :, None, None])

def load(name):
    return norm(np.load(os.path.join(dpath, name + '_x.npy'))), torch.Tensor(np.load(os.path.join(dpath, name + '_y.npy'))).long()
val_x, val_y = load('val')
test_x, test_y = load('test')

model = MNISTColorNet().to(dev)
model.load_state_dict(torch.load('mnist/ColorMNIST/orig_model_colorMNIST.pt', map_location=dev))
model.eval()

# contrast pairs: MNIST train digits, coloured with their (biased) train colour vs plain grayscale (white digit)
mnist = datasets.MNIST(root='data', train=True, download=False)
idx = np.random.choice(len(mnist), opt.n_pairs, replace=False)
digits = mnist.data[idx].numpy().astype(np.float32)[:, None]            # N,1,28,28 in [0,255]
labels = mnist.targets[idx].numpy()
x_col = norm(digits * np.stack([colors[k] for k in labels])[:, :, None, None])
x_gray = norm(np.repeat(digits, 3, axis=1))

def get_conv2(x):
    cache = {}
    h = model.conv2.register_forward_hook(lambda m, i, o: cache.__setitem__('a', o))
    model(x.to(dev))
    h.remove()
    return cache['a']

act_col, act_gray = get_conv2(x_col), get_conv2(x_gray)
diffs = {
    'actadd_single_pair': (act_col[0] - act_gray[0]),                   # exactly sv.ipynb: one prompt pair
    'actadd_mean_diff': (act_col - act_gray).mean(0),                   # averaged over n_pairs
}
g = torch.Generator(device='cpu').manual_seed(opt.seed)
rnd = torch.randn(diffs['actadd_mean_diff'].shape, generator=g).to(dev)
diffs['random_same_norm'] = rnd / rnd.norm() * diffs['actadd_mean_diff'].norm()

def evaluate(x, y, hook=None, bs=2000):
    h = model.conv2.register_forward_hook(hook) if hook else None
    correct = 0
    for i in range(0, len(x), bs):
        correct += (model(x[i:i+bs].to(dev)).argmax(1).cpu() == y[i:i+bs]).sum().item()
    if h: h.remove()
    return 100. * correct / len(x)

def add_hook(diff, coeff):
    def hook(m, i, o):
        return o + coeff * diff                                          # resid_pre += coeff * act_diff
    return hook

def proj_hook(v):
    # P-ClArC-style: remove the component along the (flattened) colour direction
    v = (v / v.norm()).flatten()
    def hook(m, i, o):
        of = o.flatten(1)
        return (of - (of @ v)[:, None] * v[None]).view_as(o)
    return hook

res = {'base': {'val': evaluate(val_x, val_y), 'test': evaluate(test_x, test_y)},
       'base_gray_test': evaluate(norm(np.repeat(np.load(os.path.join(dpath, 'test_x.npy')).max(1, keepdims=True), 3, 1)
                                       / np.stack([colors[9 - k].max() for k in test_y.numpy()])[:, None, None, None]), test_y)}
print('base', res['base'], 'base on grayscale test digits', res['base_gray_test'])

coeffs = [-20, -15, -10, -8, -6, -5, -4, -3, -2, -1.5, -1, -0.5, 0.5, 1, 2]
for name, d in diffs.items():
    sweep = {c: evaluate(val_x, val_y, add_hook(d, c)) for c in coeffs}
    best_c = max(sweep, key=sweep.get)
    res[name] = {'val_sweep': sweep, 'best_coeff_on_val': best_c, 'val': sweep[best_c],
                 'test': evaluate(test_x, test_y, add_hook(d, best_c)),
                 'test_sweep': {c: evaluate(test_x, test_y, add_hook(d, c)) for c in coeffs}}
    print(name, 'best c (val) =', best_c, 'val', sweep[best_c], 'test', res[name]['test'])

res['projection_mean_diff'] = {'val': evaluate(val_x, val_y, proj_hook(diffs['actadd_mean_diff'])),
                               'test': evaluate(test_x, test_y, proj_hook(diffs['actadd_mean_diff']))}
print('projection', res['projection_mean_diff'])
json.dump(res, open(opt.out, 'w'), indent=1)
