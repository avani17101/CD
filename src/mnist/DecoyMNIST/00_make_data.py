import os

import numpy as np
import tensorflow as tf
from colour import Color
from tqdm import tqdm

DATA_PATH = "../../data/DecoyMNIST"
os.makedirs(DATA_PATH, exist_ok=True) 

np.random.seed(0)

# red = Color("red")
# colors = list(red.range_to(Color("purple"),10))
# colors = [np.asarray(x.get_rgb()) for x in colors]

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# num_samples = len(x_train)
# color_x = np.zeros((num_samples, 3, 28, 28), dtype = np.float32)
# color_y = np.empty(num_samples, dtype = np.int16)
# for i in tqdm(range(num_samples)):
#     my_color  = colors[int(y_train[i])]
#     color_x[i] = x_train[i].astype(np.float32)[np.newaxis] * my_color[:, None, None]
#     color_y[i] = y_train[i]
# np.save(os.path.join(DATA_PATH, "train_x.npy"), color_x)
# print('train_x shape', color_x.shape)
# np.save(os.path.join(DATA_PATH, "train_y.npy"), color_y)
# print('train_y shape', color_y.shape)

# num_samples = len(x_test)
# color_x = np.zeros((num_samples, 3, 28, 28), dtype = np.float32)
# color_y = y_test.copy()
# for i in tqdm(range(num_samples)):
#     color_x[i] = x_test[i].astype(np.float32)[np.newaxis] * colors[9 - color_y[i]][:, None, None]
# np.save(os.path.join(DATA_PATH, "test_x.npy"),  color_x)
# print('test_x shape', color_x.shape)
# np.save(os.path.join(DATA_PATH, "test_y.npy"), color_y)
# print('test_y shape', color_y.shape)

color_x = np.zeros((len(x_train), 1, 28, 28))
color_x = x_train[:, None].astype(np.float32)
color_y = y_train.copy()
choice_1 = np.random.choice(2, size = len(color_x))*23
choice_2 = np.random.choice(2, size = len(color_x))*23
for i in range(len(color_x)):
    color_x[i, :, choice_1[i]:choice_1[i]+5, choice_2[i]:choice_2[i]+5] = 255 - 25*color_y[i]
color_x /= color_x.max()
color_x = color_x*2 - 1
np.save(os.path.join(DATA_PATH, 'train_x_decoy.npy'), color_x)
print('train_x_decoy shape:', color_x.shape)
np.save(os.path.join(DATA_PATH, 'train_y.npy'), color_y)
print('train_y shape', color_y.shape)

color_x_non_decoy = x_test.copy()
color_x_non_decoy = np.expand_dims(color_x_non_decoy, axis=1).astype(np.float32)
color_x = np.zeros((len(x_test), 1, 28, 28))
color_x = x_test[:, None].astype(np.float32)
color_y = y_test.copy()
choice_1 = np.random.choice(2, size = len(color_x))*23
choice_2 = np.random.choice(2, size = len(color_x))*23
for i in range(len(color_x)):
    color_x[i, :, choice_1[i]:choice_1[i]+5, choice_2[i]:choice_2[i]+5] = 0 + 25*color_y[i]
color_x /= color_x.max()
color_x = color_x*2 - 1
np.save(os.path.join(DATA_PATH, 'test_x.npy'), color_x_non_decoy)
print('test_x shape:', color_x_non_decoy.shape)
np.save(os.path.join(DATA_PATH, 'test_x_decoy.npy'), color_x)
print('test_x_decoy shape:', color_x.shape)
np.save(os.path.join(DATA_PATH, 'test_y.npy'), color_y)
print('test_y shape', color_y.shape)