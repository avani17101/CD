
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from scipy.stats import ttest_ind
import numpy as np
import sys
sys.path.append("../")
import matplotlib.pyplot as plt
import os.path
import pickle
import numpy as np
from six.moves import range
from sklearn import linear_model, svm
from sklearn import metrics
from sklearn.model_selection import train_test_split
from .tcav_utils import *
# import tensorflow as tf
import yaml
import numpy as np
import PIL
import matplotlib.pyplot as plt
import cv2
import torch
import os
import torch.nn.functional as F
import torch.nn as nn
import os
from os import listdir
from os.path import isfile, join
import cv2
from io import StringIO 
PIL.LOAD_TRUNCATED_IMAGES = True

# Metrics/Indexes
# from skimage.measure import compare_psnr

from skimage.metrics import peak_signal_noise_ratio as psnr_measure


from functools import partial
import numpy as np


class Bandwise(object):
    def __init__(self, index_fn):
        self.index_fn = index_fn

    def __call__(self, X, Y):
        C = X.shape[-1]
        bwindex = []
        for ch in range(C):
            x = X[..., ch]
            y = Y[..., ch]
            index = self.index_fn(x, y)
            bwindex.append(index)
        return bwindex




os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
def ssq_error(correct, estimate):
    """Compute the sum-squared-error for an image, where the estimate is
    multiplied by a scalar which minimizes the error. Sums over all pixels
    where mask is True. If the inputs are color, each color channel can be
    rescaled independently."""
    assert correct.ndim == 2
    if np.sum(estimate**2) > 1e-5:
        alpha = np.sum(correct * estimate) / np.sum(estimate**2)
    else:
        alpha = 0.
    return np.sum((correct - alpha*estimate) ** 2)

def local_error(correct, estimate, window_size, window_shift):
    """Returns the sum of the local sum-squared-errors, where the estimate may
    be rescaled within each local region to minimize the error. The windows are
    window_size x window_size, and they are spaced by window_shift."""
    M, N, C = correct.shape
    ssq = total = 0.
    for c in range(C):
        for i in range(0, M - window_size + 1, window_shift):
            for j in range(0, N - window_size + 1, window_shift):
                correct_curr = correct[i:i+window_size, j:j+window_size, c]
                estimate_curr = estimate[i:i+window_size, j:j+window_size, c]
                ssq += ssq_error(correct_curr, estimate_curr)
                total += np.sum(correct_curr**2)
    # assert np.isnan(ssq/total)
    return ssq / total

def srgb_to_rgb(srgb):
    ret = np.zeros_like(srgb)
    idx0 = srgb <= 0.04045
    idx1 = srgb > 0.04045
    ret[idx0] = srgb[idx0] / 12.92
    ret[idx1] = np.power((srgb[idx1] + 0.055) / 1.055, 2.4)
    return ret
    
def load_mask_from_file(filename, shape,convert_srgb=False):
    try:
        # ensure image has no transparency channel
        img = cv2.imread(filename,0)
        img = cv2.resize(img, (256,256)) 
        # Normalize pixel values to between 0 and 1.
    #         img = np.float32(img) / 255.0
    #         img = np.expand_dims(img,axis=2)
    #         img = np.concatenate((img,img,img),axis=2)
        
        return img
        # if convert_srgb:
        #     return srgb_to_rgb(img)
        # else:
        #     return img

    except Exception as e:
        print("eror in reading img", filename)
        return []
    

def load_image_from_file(filename, shape=(384,512),convert_srgb=False):
    # ensure image has no transparency channel
    try:
      # img = np.array(PIL.Image.open(open(filename, 'rb')).convert('RGB')).resize(shape, PIL.Image.BILINEAR)[:,:,:3]
      img = plt.imread(filename)
      img = cv2.resize(img, shape)[:,:,:3]
      # Normalize pixel values to between 0 and 1.
      img = np.float32(img)/ 255.0
      if convert_srgb:
          return srgb_to_rgb(img)
      else:
          return img
    except Exception as e:
        print("eror in reading img", filename)
        return []
      

def load_gray_image_from_file(filename, shape,convert_srgb=False):
    try:
        # ensure image has no transparency channel
        img = cv2.cvtColor(cv2.imread(filename),cv2.COLOR_BGR2GRAY)   
        img = cv2.resize(img, shape) 
        # Normalize pixel values to between 0 and 1.
        img = np.float32(img) / 255.0
        return img
        # if convert_srgb:
        #     return srgb_to_rgb(img)
        # else:
        #     return img

    except Exception as e:
        print("eror in reading img", filename)
        return None
    return img

def get_random_pairs(random_concepts,num='10n'):
  pairs = []
  if num=='10n':
    #10n random pairs
    
    for i,concept in enumerate(random_concepts):
        random_con = np.random.choice(random_concepts)
        random_con2 = np.random.choice(random_concepts)
        random_con3 = np.random.choice(random_concepts)
        random_con4 = np.random.choice(random_concepts)
        random_con5 = np.random.choice(random_concepts)
        random_con6 = np.random.choice(random_concepts)
        random_con7 = np.random.choice(random_concepts)
        random_con8 = np.random.choice(random_concepts)
        random_con9 = np.random.choice(random_concepts)
        random_con10 = np.random.choice(random_concepts)

        if random_con!= concept:
            pairs.append((concept,random_con))
        if random_con2!= concept:
            pairs.append((random_con2,concept))
        if random_con3!=concept:
            pairs.append((random_con3,concept))
        if random_con4!=concept:
            pairs.append((concept,random_con4))
        if random_con5!= concept:
            pairs.append((random_con5,concept))
        if random_con6!=concept:
            pairs.append((random_con6,concept))
        if random_con7!=concept:
            pairs.append((concept,random_con7))
        if random_con8!=concept:
            pairs.append((random_con8,concept))
        if random_con9!=concept:
            pairs.append((concept,random_con9))
        if random_con10!=concept:
            pairs.append((concept,random_con10))
    else:
      for i,concept in enumerate(random_concepts):
        for random_con in random_concepts[i:]:
            if random_con!= concept:
                pairs.append((concept,random_con))
                pairs.append((random_con,concept))


  return pairs
      

def get_examples_and_gt_for_input(root,shading='gray',shape=(256,256),convert_srgb=False):
    path = root+'/MIT-intrinsic/data/test/'
    img_folders = os.listdir(path)
    input_imgs = []
    mask_imgs = []
    ref_imgs = []
    sh_imgs = []
    
    for fol in img_folders:
        im = load_image_from_file(os.path.join(os.path.join(path,fol),'original.png'), shape=shape)
        im = np.ascontiguousarray(np.transpose(im,(2,0,1)))
        input_imgs.append(im)
        ref = load_image_from_file(os.path.join(os.path.join(path,fol),'reflectance.png'), shape=shape,convert_srgb=convert_srgb)
        ref = np.ascontiguousarray(np.transpose(ref,(2,0,1)))
        ref_imgs.append(ref)
        mask = load_mask_from_file(os.path.join(os.path.join(path,fol),'mask.png'), shape=shape,convert_srgb=False)
        mask_imgs.append(mask)
        if shading=='gray':
          sh = load_gray_image_from_file(os.path.join(os.path.join(path,fol),'shading.png'), shape=shape)
          sh = np.ascontiguousarray(sh)
          sh_imgs.append(sh)
        else:
          sh = load_image_from_file(os.path.join(os.path.join(path,fol),'shading.png'), shape=shape,convert_srgb=convert_srgb)
          sh = np.ascontiguousarray(np.transpose(sh,(2,0,1)))
          sh_imgs.append(sh)
  

    arap_path = root+'/ARAP/'
    ipath = arap_path+'input/'
    rpath = arap_path+'reflectance/'
    spath = arap_path+'shading/'
    mpath = arap_path+'mask/'

    ifiles = sorted([f for f in listdir(ipath) if isfile(join(ipath, f))])
    rfiles = sorted([f for f in listdir(rpath) if isfile(join(rpath, f))])
    sfiles = sorted([f for f in listdir(spath) if isfile(join(spath, f))])
    mfiles = sorted([f for f in listdir(mpath) if isfile(join(mpath, f))])
    for i,f in enumerate(ifiles):
        scene_name = f.replace('.jpg','').replace('.png','')
        mfile_name = None
        
        if scene_name+'_mask'+'.png' in mfiles:
            mfile_name = scene_name+'_mask'+'.png'
        elif scene_name+'_mask'+'.jpg' in mfiles:
            mfile_name = scene_name+'_mask'+'.jpg'
        
        if mfile_name:   
            mask =  load_mask_from_file(os.path.join(mpath,mfile_name), shape=shape)
            mask_imgs.append(mask)
        else:
            # print("creating mask of ones for scene", scene_name)
            mask = np.ones((shape[0],shape[1]))
            mask_imgs.append(mask)

            
        im = np.transpose(load_image_from_file(os.path.join(ipath,f), shape=shape,convert_srgb=convert_srgb),(2, 0, 1)) #input is always srgb for all 3 models
        input_imgs.append(im)
        
        rfile_name = None
        if scene_name+'_albedo'+'.png' in rfiles:
            rfile_name = scene_name+'_albedo'+'.png'
        elif scene_name+'_albedo'+'.jpg' in rfiles:
            rfile_name = scene_name+'_albedo'+'.jpg'
        ref = load_image_from_file(os.path.join(rpath,rfile_name), shape=shape,convert_srgb=convert_srgb)
        mask3ch = np.expand_dims(mask,axis=2)
        mask3ch = np.concatenate((mask3ch,mask3ch,mask3ch),axis=2)
        ref = ref*mask3ch
        ref = np.transpose(ref,(2, 0, 1))
        ref_imgs.append(ref)
        
        sfile_name = None
        if scene_name+'_shading'+'.png' in sfiles:
            sfile_name = scene_name+'_shading'+'.png'
        elif scene_name+'_shading'+'.jpg' in sfiles:
            sfile_name = scene_name+'_shading'+'.jpg'
            
        if shading=='gray':
          
          sh = load_gray_image_from_file(os.path.join(spath,sfile_name), shape=shape)
          sh = sh*mask
          sh_imgs.append(sh)
        else:
          sh = load_image_from_file(os.path.join(spath,sfile_name), shape=shape,convert_srgb=convert_srgb)
          sh = sh*mask3ch
          sh = np.transpose(sh,(2,0,1))
          sh_imgs.append(sh)
    
    ref_imgs = torch.FloatTensor(ref_imgs)
    ref_imgs.requires_grad=True
    sh_imgs = torch.FloatTensor(sh_imgs)
    sh_imgs.requires_grad=True
    ref_imgs = torch.FloatTensor(ref_imgs)
    # for m in mask_imgs:
    #   print(m.shape)
    mask_imgs = torch.FloatTensor(mask_imgs)
    mask_imgs.requires_grad=True
    assert(len(mask_imgs)== len(ref_imgs))
    assert(len(mask_imgs)== len(sh_imgs))
    assert(len(mask_imgs)==len(input_imgs))
    return np.array(input_imgs), {"reflectance":ref_imgs,"shading":sh_imgs,"mask":mask_imgs}

def get_eval_set():
  with open('eval_data.pkl','rb') as f:
    dic = pickle.load(f)
    input_imgs = dic['input_imgs']
    ref_imgs = dic['reflectance']
    sh_imgs = dic['shading']
    mask_imgs = dic['mask']
    ref_imgs = torch.FloatTensor(ref_imgs)
    ref_imgs.requires_grad=True
    sh_imgs = torch.FloatTensor(sh_imgs)
    sh_imgs.requires_grad=True
    ref_imgs = torch.FloatTensor(ref_imgs)

    mask_imgs = torch.FloatTensor(mask_imgs)
    mask_imgs.requires_grad=True
    assert(len(mask_imgs)== len(ref_imgs))
    assert(len(mask_imgs)== len(sh_imgs))
    assert(len(mask_imgs)==len(input_imgs))
    return np.array(input_imgs), {"reflectance":ref_imgs,"shading":sh_imgs,"mask":mask_imgs}

def get_eval_set2():
  with open('eval_data2.pkl','rb') as f:
    dic = pickle.load(f)
    input_imgs = dic['input_imgs']
    ref_imgs = dic['reflectance']
    sh_imgs = dic['shading']
    mask_imgs = dic['mask']
    ref_imgs = torch.FloatTensor(ref_imgs)
    ref_imgs.requires_grad=True
    sh_imgs = torch.FloatTensor(sh_imgs)
    sh_imgs.requires_grad=True
    ref_imgs = torch.FloatTensor(ref_imgs)
    mask_imgs = torch.FloatTensor(mask_imgs)
    mask_imgs.requires_grad=True
    assert(len(mask_imgs)== len(ref_imgs))
    assert(len(mask_imgs)== len(sh_imgs))
    assert(len(mask_imgs)==len(input_imgs))
    return np.array(input_imgs), {"reflectance":ref_imgs,"shading":sh_imgs,"mask":mask_imgs}

def get_eval_set(f_name):
  with open(f_name,'rb') as f:
    dic = pickle.load(f)
    input_imgs = dic['input_imgs']
    ref_imgs = dic['reflectance']
    sh_imgs = dic['shading']
    sh = np.array(sh_imgs)
    mask_imgs = dic['mask']
    ref_imgs = torch.FloatTensor(ref_imgs)
    ref_imgs.requires_grad=True
    sh_imgs = torch.FloatTensor(sh_imgs)
    sh_imgs.requires_grad=True
    ref_imgs = torch.FloatTensor(ref_imgs)

    mask_imgs = torch.FloatTensor(mask_imgs)
    mask_imgs.requires_grad=True
    assert(len(mask_imgs)== len(ref_imgs))
    assert(len(mask_imgs)== len(sh_imgs))
    assert(len(mask_imgs)==len(input_imgs))
    return np.array(input_imgs), {"reflectance":ref_imgs,"shading":sh_imgs,"mask":mask_imgs}

def get_eval_set3():
  with open('eval_data3.pkl','rb') as f:
    dic = pickle.load(f)
    input_imgs = dic['input_imgs']
    ref_imgs = dic['reflectance']
    sh_imgs = dic['shading']
    mask_imgs = dic['mask']
    ref_imgs = torch.FloatTensor(ref_imgs)
    ref_imgs.requires_grad=True
    sh_imgs = torch.FloatTensor(sh_imgs)
    sh_imgs.requires_grad=True
    ref_imgs = torch.FloatTensor(ref_imgs)

    mask_imgs = torch.FloatTensor(mask_imgs)
    mask_imgs.requires_grad=True
    assert(len(mask_imgs)== len(ref_imgs))
    assert(len(mask_imgs)== len(sh_imgs))
    assert(len(mask_imgs)==len(input_imgs))
    return np.array(input_imgs), {"reflectance":ref_imgs,"shading":sh_imgs,"mask":mask_imgs}

def get_eval_set_arap(typee='srgb_rgb'):
  f_name = None
  if typee == 'srgb_rgb': #input is srgb, output of model(hence GT R) is rgb
      f_name = 'eval_data_arapnew.pkl'
  else:
      f_name = 'eval_data_arapnew_srgb_srgb.pkl'
  with open(f_name,'rb') as f:
    dic = pickle.load(f)
    input_imgs = list(dic['input_imgs'])
    ref_imgs = list(dic['reflectance'])
    sh_imgs = list(dic['shading'])
    mask_imgs = list(dic['mask'])
    ref_imgs = torch.FloatTensor(np.array(ref_imgs))
    ref_imgs.requires_grad=True
    sh_imgs = torch.FloatTensor(np.array(sh_imgs))
    sh_imgs.requires_grad=True
    

    mask_imgs = torch.FloatTensor(np.array(mask_imgs))
    mask_imgs.requires_grad=True
    assert(len(mask_imgs)== len(ref_imgs))
    assert(len(mask_imgs)== len(sh_imgs))
    assert(len(mask_imgs)==len(input_imgs))
    return np.array(input_imgs), {"reflectance":ref_imgs,"shading":sh_imgs,"mask":mask_imgs}

class CAV(object):
  """CAV class contains methods for concept activation vector (CAV).

  CAV represents semenatically meaningful vector directions in
  network's embeddings (bottlenecks).
  """

  @staticmethod
  def default_hparams():
    """HParams used to train the CAV.

    you can use logistic regression or linear regression, or different
    regularization of the CAV parameters.

    Returns:
      TF.HParams for training.
    """
    return {'model_type':'linear', 'alpha':.01, 'max_iter':1000, 'tol':1e-3}

  @staticmethod
  def load_cav(cav_path):
    """Make a CAV instance from a saved CAV (pickle file).

    Args:
      cav_path: the location of the saved CAV

    Returns:
      CAV instance.
    """
    with open(cav_path, 'rb') as pkl_file:
      save_dict = pickle.load(pkl_file)

    cav = CAV(save_dict['concepts'], save_dict['bottleneck'],
              save_dict['hparams'], save_dict['saved_path'])
    cav.accuracies = save_dict['accuracies']
    cav.cavs = save_dict['cavs']
    return cav

  @staticmethod
  def cav_key(concepts, bottleneck, model_type, alpha):
    """A key of this cav (useful for saving files).

    Args:
      concepts: set of concepts used for CAV
      bottleneck: the bottleneck used for CAV
      model_type: the name of model for CAV
      alpha: a parameter used to learn CAV

    Returns:
      a string cav_key
    """
    return '-'.join([str(c) for c in concepts
                    ]) + '-' + bottleneck + '-' + model_type + '-' + str(alpha)

  @staticmethod
  def check_cav_exists(cav_dir, concepts, bottleneck, cav_hparams):
    """Check if a CAV is saved in cav_dir.

    Args:
      cav_dir: where cav pickles might be saved
      concepts: set of concepts used for CAV
      bottleneck: the bottleneck used for CAV
      cav_hparams: a parameter used to learn CAV

    Returns:
      True if exists, False otherwise.
    """
    cav_path = os.path.join(
        cav_dir,
        CAV.cav_key(concepts, bottleneck, cav_hparams['model_type'],
                    cav_hparams['alpha']) + '.pkl')
    return os.path.exists(cav_path)

  @staticmethod
  def _create_cav_training_set(concepts, bottleneck, acts):
    """Flattens acts, make mock-labels and returns the info.

    Labels are assigned in the order that concepts exists.

    Args:
        concepts: names of concepts
        bottleneck: the name of bottleneck where acts come from
        acts: a dictionary that contains activations
    Returns:
        x -  flattened acts
        labels - corresponding labels (integer)
        labels2text -  map between labels and text.
    """

    x = []
    labels = []
    labels2text = {}
    # to make sure postiive and negative examples are balanced,
    # truncate all examples to the size of the smallest concept.
    # min_data_points = np.min(
    #     [acts[concept].shape[0] for concept in acts.keys()])
    pt = []
    for concept in concepts:
      try:
        if isinstance(acts[concept], dict):
            pt.append(acts[concept][bottleneck].shape[0])
        else:
            pt.append(acts[concept].shape[0])
           
      except Exception as e:
        print(e)
        import pdb
        pdb.set_trace()
        print(concept)
    # for concept in acts.keys():
    #   try:
    #     pt.append(acts[concept].shape[0])
    #   except:
    #     continue
    # print("**************************pt",pt)
    min_data_points = np.min(np.array(pt))


    for i, concept in enumerate(concepts):
        if isinstance(acts[concept], dict):
            ac = acts[concept][bottleneck][:min_data_points].detach().cpu().numpy().reshape(min_data_points, -1)
        else:
            ac = acts[concept][:min_data_points].detach().cpu().numpy().reshape(min_data_points, -1)
        x.extend(ac)
        labels.extend([i] * min_data_points)
        labels2text[i] = concept
    x = np.array(x)
    labels = np.array(labels).astype(int)
    return x, labels, labels2text

  def __init__(self, concepts, bottleneck, hparams, save_path=None):
    """Initialize CAV class.

    Args:
      concepts: set of concepts used for CAV
      bottleneck: the bottleneck used for CAV
      hparams: a parameter used to learn CAV
      save_path: where to save this CAV
    """
    self.concepts = concepts
    self.bottleneck = bottleneck
    self.hparams = hparams
    self.save_path = save_path

  def train(self, acts):
    """Train the CAVs from the activations.

    Args:
      acts: is a dictionary of activations. In particular, acts takes for of
            {'concept1':{'bottleneck name1':[...act array...],
                         'bottleneck name2':[...act array...],...
             'concept2':{'bottleneck name1':[...act array...],
    Raises:
      ValueError: if the model_type in hparam is not compatible.
    """

    #tf.compat.v1.logging.info('training with alpha={}'.format(self.hparams['alpha']))
    x, labels, labels2text = CAV._create_cav_training_set(
        self.concepts, self.bottleneck, acts)
    if self.hparams['model_type'] == 'linear':
      lm = linear_model.SGDClassifier(alpha=self.hparams['alpha'], max_iter=self.hparams['max_iter'], tol=self.hparams['tol'])
    elif self.hparams['model_type'] == 'logistic':
      lm = linear_model.LogisticRegression()
    elif self.hparams['model_type'] == 'svm':
      # print("using svm")
      lm = svm.SVC(kernel='linear')
    else:
      raise ValueError('Invalid hparams.model_type: {}'.format(
          self.hparams['model_type']))
    
    # print("here",x, len(labels), len(labels2text))
    self.accuracies = self._train_lm(lm, x, labels, labels2text)
    if len(lm.coef_) == 1:
      # if there were only two labels, the concept is assigned to label 0 by
      # default. So we flip the coef_ to reflect this.
      self.cavs = [-1 * lm.coef_[0], lm.coef_[0]]
    else:
      self.cavs = [c for c in lm.coef_]
    self._save_cavs()

  def perturb_act(self, act, concept, operation=np.add, alpha=1.0):
    """Make a perturbation of act with a direction of this CAV.

    Args:
      act: activations to be perturbed
      concept: the concept to perturb act with.
      operation: the operation will be ran to perturb.
      alpha: size of the step.

    Returns:
      perturbed activation: same shape as act
    """
    flat_act = np.reshape(act, -1)
    pert = operation(flat_act, alpha * self.get_direction(concept))
    return np.reshape(pert, act.shape)

  def get_key(self):
    """Returns cav_key."""

    return CAV.cav_key(self.concepts, self.bottleneck, self.hparams['model_type'],
                       self.hparams['alpha'])

  def get_direction(self, concept):
    """Get CAV direction.

    Args:
      concept: the conept of interest

    Returns:
      CAV vector.
    """
    return self.cavs[self.concepts.index(concept)]

  def _save_cavs(self):
    """Save a dictionary of this CAV to a pickle."""
    save_dict = {
        'concepts': self.concepts,
        'bottleneck': self.bottleneck,
        'hparams': self.hparams,
        'accuracies': self.accuracies,
        'cavs': self.cavs,
        'saved_path': self.save_path
    }
    if self.save_path is not None:
      with open(self.save_path, 'wb') as pkl_file:
        pickle.dump(save_dict, pkl_file)
    # else:
    #   #tf.compat.v1.logging.info('save_path is None. Not saving anything')

  def _train_lm(self, lm, x, y, labels2text):
    """Train a model to get CAVs.

    Modifies lm by calling the lm.fit functions. The cav coefficients are then
    in lm._coefs.

    Args:
      lm: An sklearn linear_model object. Can be linear regression or
        logistic regression. Must support .fit and ._coef.
      x: An array of training data of shape [num_data, data_dim]
      y: An array of integer labels of shape [num_data]
      labels2text: Dictionary of text for each label.

    Returns:
      Dictionary of accuracies of the CAVs.

    """
#     print(x[0], y[0])
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.33, stratify=y)
    # print(len(x_train),len(y_train))
    # if you get setting an array element with a sequence, chances are that your
    # each of your activation had different shape - make sure they are all from
    # the same layer, and input image size was the same
    # print("x_train, y_train ",x_train, y_train)
    lm.fit(x_train, y_train)
    y_pred = lm.predict(x_test)
    # get acc for each class.
    num_classes = max(y) + 1
    acc = {}
    num_correct = 0
    for class_id in range(num_classes):
        # get indices of all test data that has this class.
        idx = (y_test == class_id)
        acc[labels2text[class_id]] = metrics.accuracy_score(
          y_pred[idx], y_test[idx])
        # overall correctness is weighted by the number of examples in this class.
        num_correct += (sum(idx) * acc[labels2text[class_id]])
    acc['overall'] = float(num_correct) / float(len(y_test))
    # print("cav sklearn acc",acc['overall'])
    #tf.compat.v1.logging.info('acc per class %s' % (str(acc)))
    return acc


def get_or_train_cav(concepts,
                     bottleneck,
                     acts,
                     cav_dir=None,
                     cav_hparams=None,
                     overwrite=True):
  """Gets, creating and training if necessary, the specified CAV.

  Assumes the activations already exists.

  Args:
    concepts: set of concepts used for CAV
            Note: if there are two concepts, provide the positive concept
                  first, then negative concept (e.g., ['striped', 'random500_1']
    bottleneck: the bottleneck used for CAV
    acts: dictionary contains activations of concepts in each bottlenecks
          e.g., acts[concept]
    cav_dir: a directory to store the results.
    cav_hparams: a parameter used to learn CAV
    overwrite: if set to True overwrite any saved CAV files.

  Returns:
    returns a CAV instance
  """

  if cav_hparams is None:
    cav_hparams = CAV.default_hparams()

  cav_path = None
  if cav_dir is not None:
    make_dir_if_not_exists(cav_dir)
    cav_path = os.path.join(
        cav_dir,
        CAV.cav_key(concepts, bottleneck, cav_hparams['model_type'],
                    cav_hparams['alpha']).replace('/', '.') + '.pkl')

    # if not overwrite and os.path.exists(cav_path):
    #   #tf.compat.v1.logging.info('CAV already exists: {}'.format(cav_path))
    #   cav_instance = CAV.load_cav(cav_path)
    #   #tf.compat.v1.logging.info('CAV accuracies: {}'.format(cav_instance.accuracies))
    #   return cav_instance

  #tf.compat.v1.logging.info('Training CAV {} - {} alpha {}'.format(
  #     concepts, bottleneck, cav_hparams['alpha']))
  cav_instance = CAV(concepts, bottleneck, cav_hparams, cav_path)
  # print("acts",acts.keys())

  cav_instance.train({c: acts[c] for c in concepts})
  #tf.compat.v1.logging.info('CAV accuracies: {}'.format(cav_instance.accuracies))
  return cav_instance



# helper function to output plot and write summary data
def plot_results(results, random_counterpart=None, random_concepts=None, num_random_exp=100,
    min_p_val=0.05,name='S',exp_name='test'):
  """Helper function to organize results.
  When run in a notebook, outputs a matplotlib bar plot of the
  TCAV scores for all bottlenecks for each concept, replacing the
  bars with asterisks when the TCAV score is not statistically significant.
  If you ran TCAV with a random_counterpart, supply it here, otherwise supply random_concepts.
  If you get unexpected output, make sure you are using the correct keywords.

  Args:
    results: dictionary of results from TCAV runs.
    random_counterpart: name of the random_counterpart used, if it was used. 
    random_concepts: list of random experiments that were run. 
    num_random_exp: number of random experiments that were run.
    min_p_val: minimum p value for statistical significance
  """

  # helper function, returns if this is a random concept
  def is_random_concept(concept):
    if random_counterpart:
      return random_counterpart == concept
    
    elif random_concepts:
      return concept in random_concepts

    else:
      return 'random500_' in concept


  # dict with keys of concepts containing dict with bottlenecks
  result_summary = {}
    
  # random
  random_i_ups = {}
    
  for result in results:
    if result['cav_concept'] not in result_summary:
      result_summary[result['cav_concept']] = {}
    
    if result['bottleneck'] not in result_summary[result['cav_concept']]:
      result_summary[result['cav_concept']][result['bottleneck']] = []
    
    result_summary[result['cav_concept']][result['bottleneck']].append(result)

    # store random
    if is_random_concept(result['cav_concept']):
      if result['bottleneck'] not in random_i_ups:
        random_i_ups[result['bottleneck']] = []
        
      random_i_ups[result['bottleneck']].append(result['i_up'])
    
  # to plot, must massage data again 
  plot_data = {}
  plot_concepts = []
    
  # print concepts and classes with indentation
  for concept in result_summary:
        
    # if not random
    if not is_random_concept(concept):
      # print(" ", "Concept =", concept)
      plot_concepts.append(concept)

      for bottleneck in result_summary[concept]:
        import pdb
        pdb.set_trace()
        i_ups = [item['i_up'] for item in result_summary[concept][bottleneck]]
        
        # Calculate statistical significance
        _, p_val = ttest_ind(random_i_ups[bottleneck], i_ups)
                  
        if bottleneck not in plot_data:
          plot_data[bottleneck] = {'bn_vals': [], 'bn_stds': [], 'significant': []}

        if p_val > min_p_val:
          # statistically insignificant
          plot_data[bottleneck]['bn_vals'].append(0.01)
          plot_data[bottleneck]['bn_stds'].append(0)
          plot_data[bottleneck]['significant'].append(False)
            
        else:
          plot_data[bottleneck]['bn_vals'].append(np.mean(i_ups))
          plot_data[bottleneck]['bn_stds'].append(np.std(i_ups))
          plot_data[bottleneck]['significant'].append(True)

        # print(3 * " ", "Bottleneck =", ("%s. TCAV Score = %.2f (+- %.2f), "
        #     "random was %.2f (+- %.2f). p-val = %.3f (%s)") % (
        #     bottleneck, np.mean(i_ups), np.std(i_ups),
        #     np.mean(random_i_ups[bottleneck]),
        #     np.std(random_i_ups[bottleneck]), p_val,
        #     "not significant" if p_val > min_p_val else "significant"))
        
  # subtract number of random experiments
  if random_counterpart:
    num_concepts = len(result_summary) - 1
  elif random_concepts:
    num_concepts = len(result_summary) - len(random_concepts)
  else: 
    num_concepts = len(result_summary) - num_random_exp
    
  num_bottlenecks = len(plot_data)
  bar_width = 0.35
    
  # create location for each bar. scale by an appropriate factor to ensure 
  # the final plot doesn't have any parts overlapping
  index = np.arange(num_concepts) * bar_width * (num_bottlenecks + 1)

  # matplotlib
  fig, ax = plt.subplots()
    
  # draw all bottlenecks individually
  for i, [bn, vals] in enumerate(plot_data.items()):
    bar = ax.bar(index + i * bar_width, vals['bn_vals'],
        bar_width, yerr=vals['bn_stds'], label=bn)
    
    # draw stars to mark bars that are stastically insignificant to 
    # show them as different from others
    for j, significant in enumerate(vals['significant']):
      if not significant:
        ax.text(index[j] + i * bar_width - 0.1, 0.01, "*",
            fontdict = {'weight': 'bold', 'size': 16,
            'color': bar.patches[0].get_facecolor()})
  # print(plot_data)
  # set propertiesq
  ax.set_title(name)
  ax.set_ylabel('TCAV Score')
  ax.set_xticks(index + num_bottlenecks * bar_width / 2)
  ax.set_xticklabels(plot_concepts)
  ax.legend()
  fig.tight_layout()
  plt.show()
  plt.savefig(exp_name+name+'.png')
  
def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream)

def gen_data():
    bunny = np.load('bunny.npy')
    cow = np.load('cow.npy')
    num_rows, num_cols = bunny.shape[:2]

    # bottom right: 46 39 to 60 70
    translation_matrix = np.float32([ [1,0,0], [0,1,0] ])
    img_translation = cv2.warpAffine(cow, translation_matrix, (num_cols, num_rows))
    plt.imshow(img_translation,cmap='gray')

    biased_train_bun = []
    biased_train_cow = []

    for i in range(1000):

      x_val = np.random.randint(46,60)
      y_val = np.random.randint(39,70)
      translation_matrix = np.float32([ [1,0,x_val], [0,1,y_val] ])
      im = cv2.warpAffine(bunny, translation_matrix, (200, 200))
      biased_train_bun.append(im)
      x_val = np.random.randint(-53,-39)
      y_val = np.random.randint(-65,-34)
      translation_matrix = np.float32([ [1,0,x_val], [0,1,y_val] ])
      im = cv2.warpAffine(bunny, translation_matrix, (200, 200))
      biased_train_bun.append(im)

      y_val = np.random.randint(44,64)
      x_val = np.random.randint(-46,-22)
      translation_matrix = np.float32([ [1,0,x_val], [0,1,y_val] ])
      im = cv2.warpAffine(cow, translation_matrix, (200, 200))
      biased_train_cow.append(im)
      y_val = np.random.randint(-56,-33)
      x_val = np.random.randint(55,78)
      translation_matrix = np.float32([ [1,0,x_val], [0,1,y_val] ])
      im = cv2.warpAffine(cow, translation_matrix, (200, 200))
      biased_train_cow.append(im)

    unbiased_train_bun = []
    unbiased_train_cow = []

    for i in range(2000):

      x_val = np.random.randint(-53,60)
      y_val = np.random.randint(-65,70)
      translation_matrix = np.float32([ [1,0,x_val], [0,1,y_val] ])
      im = cv2.warpAffine(bunny, translation_matrix, (200, 200))
      unbiased_train_bun.append(im)

      y_val = np.random.randint(-56,64)
      x_val = np.random.randint(-46,78)
      translation_matrix = np.float32([ [1,0,x_val], [0,1,y_val] ])
      im = cv2.warpAffine(cow, translation_matrix, (200, 200))
      unbiased_train_cow.append(im)

    biased_train_cow = np.array(biased_train_cow)
    biased_train_bun = np.array(biased_train_bun)
    unbiased_train_cow = np.array(unbiased_train_cow)
    unbiased_train_bun = np.array(unbiased_train_bun)


    """# New Section"""

    unbiased_squares = unbiased_train_cow
    unbiased_hearts = unbiased_train_bun
    biased_squares = biased_train_cow
    biased_hearts = biased_train_bun

    some_unbiased_squares = biased_squares[:25]
    np.random.shuffle(some_unbiased_squares)

    # grid = make_grid(torch.from_numpy(np.float32(some_unbiased_squares)).view(25,1,200,200),nrow=5,pad_value=1,padding=10)
    # plt.axis('off')
    # plt.imsave('biased_stars.png',grid.numpy().transpose(1,2,0))
    # plt.imshow(grid.numpy().transpose(1,2,0))


    unbiased_data = np.concatenate((unbiased_squares,unbiased_hearts))
    labels = np.concatenate((np.zeros(len(unbiased_squares)),np.ones(len(unbiased_squares))))
    biased_data = np.concatenate((biased_squares,biased_hearts))

    np.save('biased_data',biased_data)
    np.save('unbiased_data',unbiased_data)
    np.save("labels",labels)
    return biased_data, unbiased_data, labels

def get_data():
    if os.path.exists('biased_data.npy') and os.path.exists('unbiased_data.npy') and os.path.exists('labels.npy'):
        return np.load('biased_data.npy'), np.load('unbiased_data.npy'), np.load('labels.npy')
    else:
        return gen_data()

def get_concept_set(concepts_key,typee='small'):
  albedo_change_concepts = ['Cube_happy_stanford-bunnylight1temp6500', 'Cube_happy_stanford-bunnylight2temp6500', 'Cube_happy_stanford-bunnylight3temp6500', 'Cube_happy_stanford-bunnytex_light0temp6500', 'Cube_happy_stanford-bunnytex_light1temp6500', 'Cube_happy_stanford-bunnytex_light2temp6500', 'Cube_happy_stanford-bunnytex_light3temp6500', 'Cubelight0temp6500', 'Cubelight1temp6500', 'Cubelight2temp6500', 'Cubelight3temp6500', 'Cubetex_light0temp6500', 'Cubetex_light1temp6500', 'Cubetex_light2temp6500', 'Cubetex_light3temp6500', 'Sphere_Cube_teapotlight0temp6500', 'Sphere_Cube_teapotlight1temp6500', 'Sphere_Cube_teapotlight2temp6500', 'Sphere_Cube_teapotlight3temp6500', 'Sphere_Cube_teapottex_light0temp6500', 'Sphere_Cube_teapottex_light1temp6500', 'Sphere_Cube_teapottex_light2temp6500', 'Sphere_Cube_teapottex_light3temp6500', 'Sphere_nefertiti_max-plancklight0temp6500', 'Sphere_nefertiti_max-plancklight1temp6500', 'Sphere_nefertiti_max-plancklight2temp6500', 'Sphere_nefertiti_max-plancklight3temp6500', 'Sphere_nefertiti_max-plancktex_light0temp6500', 'Sphere_nefertiti_max-plancktex_light1temp6500', 'Sphere_nefertiti_max-plancktex_light2temp6500', 'Sphere_nefertiti_max-plancktex_light3temp6500', 'Spherelight0temp6500', 'Spherelight1temp6500', 'Spherelight2temp6500', 'Spherelight3temp6500', 'Spheretex_light0temp6500', 'Spheretex_light1temp6500', 'Spheretex_light2temp6500', 'Spheretex_light3temp6500', 'happy_homer_Spherelight0temp6500', 'happy_homer_Spherelight1temp6500', 'happy_homer_Spherelight2temp6500', 'happy_homer_Spherelight3temp6500', 'happy_homer_Spheretex_light0temp6500', 'happy_homer_Spheretex_light1temp6500', 'happy_homer_Spheretex_light2temp6500', 'happy_homer_Spheretex_light3temp6500', 'happy_homer_max-plancklight0temp6500', 'happy_homer_max-plancklight1temp6500', 'happy_homer_max-plancklight2temp6500', 'happy_homer_max-plancklight3temp6500', 'happy_homer_max-plancktex_light0temp6500', 'happy_homer_max-plancktex_light1temp6500', 'happy_homer_max-plancktex_light2temp6500', 'happy_homer_max-plancktex_light3temp6500', 'happy_max-planck_spotlight0temp6500', 'happy_max-planck_spotlight1temp6500', 'happy_max-planck_spotlight2temp6500', 'happy_max-planck_spotlight3temp6500', 'happy_max-planck_spottex_light0temp6500', 'happy_max-planck_spottex_light1temp6500', 'happy_max-planck_spottex_light2temp6500', 'happy_max-planck_spottex_light3temp6500', 'happylight0temp6500', 'happylight1temp6500', 'happylight2temp6500', 'happylight3temp6500', 'happytex_light0temp6500', 'happytex_light1temp6500', 'happytex_light2temp6500', 'happytex_light3temp6500', 'homer_max-planck_suzannelight0temp6500', 'homer_max-planck_suzannelight1temp6500', 'homer_max-planck_suzannelight2temp6500', 'homer_max-planck_suzannelight3temp6500', 'homer_max-planck_suzannetex_light0temp6500', 'homer_max-planck_suzannetex_light1temp6500', 'homer_max-planck_suzannetex_light2temp6500', 'homer_max-planck_suzannetex_light3temp6500', 'homerlight0temp6500', 'homerlight1temp6500', 'homerlight2temp6500', 'homerlight3temp6500', 'homertex_light0temp6500', 'homertex_light1temp6500', 'homertex_light2temp6500', 'homertex_light3temp6500', 'max-planck_stanford-bunny_teapotlight0temp6500', 'max-planck_stanford-bunny_teapotlight1temp6500', 'max-planck_stanford-bunny_teapotlight2temp6500', 'max-planck_stanford-bunny_teapotlight3temp6500', 'max-planck_stanford-bunny_teapottex_light0temp6500', 'max-planck_stanford-bunny_teapottex_light1temp6500', 'max-planck_stanford-bunny_teapottex_light2temp6500', 'max-planck_stanford-bunny_teapottex_light3temp6500', 'max-plancklight0temp6500', 'max-plancklight1temp6500', 'max-plancklight2temp6500', 'max-plancklight3temp6500', 'max-plancktex_light0temp6500', 'max-plancktex_light1temp6500', 'max-plancktex_light2temp6500', 'max-plancktex_light3temp6500', 'nefertiti_spot_suzannelight0temp6500', 'nefertiti_spot_suzannelight1temp6500', 'nefertiti_spot_suzannelight2temp6500', 'nefertiti_spot_suzannelight3temp6500', 'nefertitilight0temp6500', 'nefertitilight1temp6500', 'nefertitilight2temp6500', 'nefertitilight3temp6500', 'nefertititex_light0temp6500', 'nefertititex_light1temp6500', 'nefertititex_light2temp6500', 'nefertititex_light3temp6500', 'spot_stanford-bunny_suzannelight0temp6500', 'spot_stanford-bunny_suzannelight1temp6500', 'spot_stanford-bunny_suzannelight2temp6500', 'spot_stanford-bunny_suzannelight3temp6500', 'spot_stanford-bunny_suzannetex_light0temp6500', 'spot_stanford-bunny_suzannetex_light1temp6500', 'spot_stanford-bunny_suzannetex_light2temp6500', 'spot_stanford-bunny_suzannetex_light3temp6500', 'spotlight0temp6500', 'spotlight1temp6500', 'spotlight2temp6500', 'spotlight3temp6500', 'spottex_light0temp6500', 'spottex_light1temp6500', 'spottex_light2temp6500', 'spottex_light3temp6500', 'stanford-bunnylight0temp6500', 'stanford-bunnylight1temp6500', 'stanford-bunnylight2temp6500', 'stanford-bunnylight3temp6500', 'stanford-bunnytex_light0temp6500', 'stanford-bunnytex_light1temp6500', 'stanford-bunnytex_light2temp6500', 'stanford-bunnytex_light3temp6500', 'suzannelight0temp6500', 'suzannelight1temp6500', 'suzannelight2temp6500', 'suzannelight3temp6500', 'suzannetex_light0temp6500', 'suzannetex_light1temp6500', 'suzannetex_light2temp6500', 'suzannetex_light3temp6500', 'teapotlight0temp6500', 'teapotlight1temp6500', 'teapotlight2temp6500', 'teapotlight3temp6500', 'teapottex_light0temp6500', 'teapottex_light1temp6500', 'teapottex_light2temp6500', 'teapottex_light3temp6500']
  ill_change_concepts = ['Cube_happy_stanford-bunny_mat0temp6500', 'Cube_happy_stanford-bunny_mat1temp6500', 'Cube_happy_stanford-bunny_mat2temp6500', 'Cube_happy_stanford-bunny_tex_mat0temp6500', 'Cube_happy_stanford-bunny_tex_mat1temp6500', 'Cube_happy_stanford-bunny_tex_mat2temp6500', 'Cube_mat0temp6500', 'Cube_mat1temp6500', 'Cube_mat2temp6500', 'Cube_tex_mat0temp6500', 'Cube_tex_mat1temp6500', 'Cube_tex_mat2temp6500', 'Sphere_Cube_teapot_mat0temp6500', 'Sphere_Cube_teapot_mat1temp6500', 'Sphere_Cube_teapot_mat2temp6500', 'Sphere_Cube_teapot_tex_mat0temp6500', 'Sphere_Cube_teapot_tex_mat1temp6500', 'Sphere_Cube_teapot_tex_mat2temp6500', 'Sphere_mat0temp6500', 'Sphere_mat1temp6500', 'Sphere_mat2temp6500', 'Sphere_nefertiti_max-planck_mat0temp6500', 'Sphere_nefertiti_max-planck_mat1temp6500', 'Sphere_nefertiti_max-planck_mat2temp6500', 'Sphere_nefertiti_max-planck_tex_mat0temp6500', 'Sphere_nefertiti_max-planck_tex_mat1temp6500', 'Sphere_nefertiti_max-planck_tex_mat2temp6500', 'Sphere_tex_mat0temp6500', 'Sphere_tex_mat1temp6500', 'Sphere_tex_mat2temp6500', 'happy_homer_Sphere_mat0temp6500', 'happy_homer_Sphere_mat1temp6500', 'happy_homer_Sphere_mat2temp6500', 'happy_homer_Sphere_tex_mat0temp6500', 'happy_homer_Sphere_tex_mat1temp6500', 'happy_homer_Sphere_tex_mat2temp6500', 'happy_homer_max-planck_mat0temp6500', 'happy_homer_max-planck_mat1temp6500', 'happy_homer_max-planck_mat2temp6500', 'happy_homer_max-planck_tex_mat0temp6500', 'happy_homer_max-planck_tex_mat1temp6500', 'happy_homer_max-planck_tex_mat2temp6500', 'happy_mat0temp6500', 'happy_mat1temp6500', 'happy_mat2temp6500', 'happy_max-planck_spot_mat0temp6500', 'happy_max-planck_spot_mat1temp6500', 'happy_max-planck_spot_mat2temp6500', 'happy_max-planck_spot_tex_mat0temp6500', 'happy_max-planck_spot_tex_mat1temp6500', 'happy_max-planck_spot_tex_mat2temp6500', 'happy_tex_mat0temp6500', 'happy_tex_mat1temp6500', 'happy_tex_mat2temp6500', 'homer_mat0temp6500', 'homer_mat1temp6500', 'homer_mat2temp6500', 'homer_max-planck_suzanne_mat0temp6500', 'homer_max-planck_suzanne_mat1temp6500', 'homer_max-planck_suzanne_mat2temp6500', 'homer_max-planck_suzanne_tex_mat0temp6500', 'homer_max-planck_suzanne_tex_mat1temp6500', 'homer_max-planck_suzanne_tex_mat2temp6500', 'homer_tex_mat0temp6500', 'homer_tex_mat1temp6500', 'homer_tex_mat2temp6500', 'max-planck_mat0temp6500', 'max-planck_mat1temp6500', 'max-planck_mat2temp6500', 'max-planck_stanford-bunny_teapot_mat0temp6500', 'max-planck_stanford-bunny_teapot_mat1temp6500', 'max-planck_stanford-bunny_teapot_mat2temp6500', 'max-planck_stanford-bunny_teapot_tex_mat0temp6500', 'max-planck_stanford-bunny_teapot_tex_mat1temp6500', 'max-planck_stanford-bunny_teapot_tex_mat2temp6500', 'max-planck_tex_mat0temp6500', 'max-planck_tex_mat1temp6500', 'max-planck_tex_mat2temp6500', 'nefertiti_mat0temp6500', 'nefertiti_mat1temp6500', 'nefertiti_mat2temp6500', 'nefertiti_spot_suzanne_mat0temp6500', 'nefertiti_spot_suzanne_mat1temp6500', 'nefertiti_spot_suzanne_mat2temp6500', 'nefertiti_tex_mat0temp6500', 'nefertiti_tex_mat1temp6500', 'nefertiti_tex_mat2temp6500', 'spot_mat0temp6500', 'spot_mat1temp6500', 'spot_mat2temp6500', 'spot_stanford-bunny_suzanne_mat0temp6500', 'spot_stanford-bunny_suzanne_mat1temp6500', 'spot_stanford-bunny_suzanne_mat2temp6500', 'spot_stanford-bunny_suzanne_tex_mat0temp6500', 'spot_stanford-bunny_suzanne_tex_mat1temp6500', 'spot_stanford-bunny_suzanne_tex_mat2temp6500', 'spot_tex_mat0temp6500', 'spot_tex_mat1temp6500', 'spot_tex_mat2temp6500', 'stanford-bunny_mat0temp6500', 'stanford-bunny_mat1temp6500', 'stanford-bunny_mat2temp6500', 'stanford-bunny_tex_mat0temp6500', 'stanford-bunny_tex_mat1temp6500', 'stanford-bunny_tex_mat2temp6500', 'suzanne_mat0temp6500', 'suzanne_mat1temp6500', 'suzanne_mat2temp6500', 'suzanne_tex_mat0temp6500', 'suzanne_tex_mat1temp6500', 'suzanne_tex_mat2temp6500', 'teapot_mat0temp6500', 'teapot_mat1temp6500', 'teapot_mat2temp6500', 'teapot_tex_mat0temp6500', 'teapot_tex_mat1temp6500', 'teapot_tex_mat2temp6500']

  albedo_change_concepts_small = ['teapotlight1temp6500', 'Cubelight2temp6500', 'nefertitilight3temp6500', 'Spherelight1temp6500', 'spot_stanford-bunny_suzannelight1temp6500', 'happy_homer_max-plancktex_light0temp6500', 'Cube_happy_stanford-bunnytex_light3temp6500', 'suzannetex_light3temp6500', 'homertex_light1temp6500', 'happy_homer_Spheretex_light1temp6500', 'happy_max-planck_spottex_light1temp6500', 'max-planck_stanford-bunny_teapotlight2temp6500', 'max-planck_stanford-bunny_teapottex_light0temp6500', 'stanford-bunnylight3temp6500', 'Cube_happy_stanford-bunnylight1temp6500', 'happytex_light2temp6500', 'suzannelight0temp6500', 'spottex_light0temp6500', 'spot_stanford-bunny_suzannetex_light2temp6500', 'happylight0temp6500', 'stanford-bunnytex_light2temp6500', 'Sphere_nefertiti_max-plancktex_light2temp6500', 'nefertititex_light2temp6500', 'homerlight1temp6500', 'happy_homer_max-plancklight0temp6500', 'happy_homer_Spherelight3temp6500', 'Sphere_Cube_teapottex_light0temp6500', 'teapottex_light2temp6500', 'Sphere_nefertiti_max-plancklight2temp6500', 'Sphere_Cube_teapotlight3temp6500', 'homer_max-planck_suzannelight2temp6500', 'happy_max-planck_spotlight2temp6500', 'max-plancktex_light1temp6500', 'nefertiti_spot_suzannelight2temp6500', 'Cubetex_light1temp6500', 'spotlight2temp6500', 'max-plancklight0temp6500', 'Spheretex_light3temp6500', 'homer_max-planck_suzannetex_light3temp6500']
  ill_change_concepts_small = ['max-planck_stanford-bunny_teapot_mat0temp6500', 'teapot_mat1temp6500', 'homer_mat1temp6500', 'spot_stanford-bunny_suzanne_mat0temp6500', 'happy_max-planck_spot_mat1temp6500', 'Cube_happy_stanford-bunny_tex_mat1temp6500', 'max-planck_mat1temp6500', 'happy_max-planck_spot_tex_mat2temp6500', 'spot_stanford-bunny_suzanne_tex_mat2temp6500', 'nefertiti_tex_mat0temp6500', 'max-planck_stanford-bunny_teapot_tex_mat2temp6500', 'happy_tex_mat0temp6500', 'nefertiti_mat1temp6500', 'Sphere_tex_mat1temp6500', 'Cube_tex_mat1temp6500', 'Sphere_nefertiti_max-planck_tex_mat1temp6500', 'max-planck_tex_mat1temp6500', 'happy_homer_max-planck_tex_mat2temp6500', 'nefertiti_spot_suzanne_mat0temp6500', 'happy_homer_max-planck_mat2temp6500', 'Sphere_Cube_teapot_tex_mat0temp6500', 'happy_mat2temp6500', 'Sphere_Cube_teapot_mat2temp6500', 'stanford-bunny_tex_mat1temp6500', 'Cube_happy_stanford-bunny_mat1temp6500', 'spot_mat1temp6500', 'stanford-bunny_mat2temp6500', 'spot_tex_mat0temp6500', 'suzanne_mat1temp6500']

  if concepts_key == 'albedo_change':
    return albedo_change_concepts_small, ill_change_concepts #pos, neg 
  else:
    return ill_change_concepts_small, albedo_change_concepts #pos, neg
    

  # if concepts_key=='albedo_neww':
  #   num_imgs = 100
  #   #concepts = ['Cubelight0', 'Cubelight1', 'Cubelight2', 'Spherelight0', 'Spherelight1', 'Spherelight2', 'happylight0', 'happylight1', 'happylight2', 'happylight4', 'homerlight0', 'homerlight1', 'homerlight2', 'homerlight4', 'max-plancklight0', 'max-plancklight1', 'max-plancklight2', 'max-plancklight4', 'nefertitilight0', 'nefertitilight1', 'nefertitilight2', 'nefertitilight4', 'spotlight0', 'spotlight1', 'spotlight2', 'spotlight4', 'stanford-bunnylight0', 'stanford-bunnylight1', 'stanford-bunnylight2', 'suzannelight0', 'suzannelight1', 'suzannelight2', 'suzannelight4', 'teapotlight0', 'teapotlight1', 'teapotlight2', 'teapotlight4']
  #   concepts = ['Cubelight0temp2500', 'Cubelight0temp4500', 'Cubelight0temp6500', 'Cubelight1temp2500', 'Cubelight1temp4500', 'Cubelight1temp6500', 'Cubelight2temp2500', 'Cubelight2temp4500', 'Cubelight2temp6500', 'Spherelight0temp2500', 'Spherelight0temp4500', 'Spherelight0temp6500', 'Spherelight1temp2500', 'Spherelight1temp4500', 'Spherelight1temp6500', 'Spherelight2temp2500', 'Spherelight2temp4500', 'Spherelight2temp6500', 'happylight0temp2500', 'happylight0temp4500', 'happylight0temp6500', 'happylight1temp2500', 'happylight1temp4500', 'happylight1temp6500', 'happylight2temp2500', 'happylight2temp4500', 'happylight2temp6500',  'homerlight0temp2500', 'homerlight0temp4500', 'homerlight0temp6500', 'homerlight1temp2500', 'homerlight1temp4500', 'homerlight1temp6500', 'homerlight2temp2500', 'homerlight2temp4500', 'homerlight2temp6500',  'max-plancklight0temp2500', 'max-plancklight0temp4500', 'max-plancklight0temp6500', 'max-plancklight1temp2500', 'max-plancklight1temp4500', 'max-plancklight1temp6500', 'max-plancklight2temp2500', 'max-plancklight2temp4500', 'max-plancklight2temp6500', 'nefertitilight0temp2500', 'nefertitilight0temp4500', 'nefertitilight0temp6500', 'nefertitilight1temp2500', 'nefertitilight1temp4500', 'nefertitilight1temp6500', 'nefertitilight2temp2500', 'nefertitilight2temp4500', 'nefertitilight2temp6500',  'spotlight0temp2500', 'spotlight0temp4500', 'spotlight0temp6500', 'spotlight1temp2500', 'spotlight1temp4500', 'spotlight1temp6500', 'spotlight2temp2500', 'spotlight2temp4500', 'spotlight2temp6500', 'stanford-bunnylight0temp2500', 'stanford-bunnylight0temp4500', 'stanford-bunnylight0temp6500', 'stanford-bunnylight1temp2500', 'stanford-bunnylight1temp4500', 'stanford-bunnylight1temp6500', 'stanford-bunnylight2temp2500', 'stanford-bunnylight2temp4500', 'stanford-bunnylight2temp6500', 'suzannelight0temp2500', 'suzannelight0temp4500', 'suzannelight0temp6500', 'suzannelight1temp2500', 'suzannelight1temp4500', 'suzannelight1temp6500', 'suzannelight2temp2500', 'suzannelight2temp4500', 'suzannelight2temp6500',  'teapotlight0temp2500', 'teapotlight0temp4500', 'teapotlight0temp6500', 'teapotlight1temp2500', 'teapotlight1temp4500', 'teapotlight1temp6500', 'teapotlight2temp2500', 'teapotlight2temp4500', 'teapotlight2temp6500']
  # elif concepts_key=='shading_neww':
  #   num_imgs = 100
  #   concepts = ['Cube_mat0temp2500', 'Cube_mat0temp4500', 'Cube_mat0temp6500', 'Cube_mat1temp2500', 'Cube_mat1temp4500', 'Cube_mat1temp6500','Cube_mat2temp2500', 'Cube_mat2temp4500', 'Cube_mat2temp6500','Sphere_mat0temp2500', 'Sphere_mat0temp4500', 'Sphere_mat0temp6500','Sphere_mat1temp2500', 'Sphere_mat1temp4500', 'Sphere_mat1temp6500','Sphere_mat2temp2500', 'Sphere_mat2temp4500', 'Sphere_mat2temp6500', 'happy_mat0temp2500', 'happy_mat0temp4500', 'happy_mat0temp6500', 'happy_mat1temp2500', 'happy_mat1temp4500', 'happy_mat1temp6500', 'happy_mat2temp2500', 'happy_mat2temp4500', 'happy_mat2temp6500', 'homer_mat0temp2500', 'homer_mat0temp4500', 'homer_mat0temp6500', 'homer_mat1temp2500', 'homer_mat1temp4500', 'homer_mat1temp6500', 'homer_mat2temp2500', 'homer_mat2temp4500', 'homer_mat2temp6500','max-planck_mat0temp2500', 'max-planck_mat0temp4500', 'max-planck_mat0temp6500', 'max-planck_mat1temp2500', 'max-planck_mat1temp4500', 'max-planck_mat1temp6500', 'max-planck_mat2temp2500', 'max-planck_mat2temp4500', 'max-planck_mat2temp6500', 'nefertiti_mat0temp2500', 'nefertiti_mat0temp4500', 'nefertiti_mat0temp6500', 'nefertiti_mat1temp2500', 'nefertiti_mat1temp4500', 'nefertiti_mat1temp6500', 'nefertiti_mat2temp2500', 'nefertiti_mat2temp4500', 'nefertiti_mat2temp6500', 'spot_mat0temp2500', 'spot_mat0temp4500', 'spot_mat0temp6500', 'spot_mat1temp2500', 'spot_mat1temp4500', 'spot_mat1temp6500', 'spot_mat2temp2500', 'spot_mat2temp4500', 'spot_mat2temp6500', 'stanford-bunny_mat0temp2500', 'stanford-bunny_mat0temp4500', 'stanford-bunny_mat0temp6500', 'stanford-bunny_mat1temp2500', 'stanford-bunny_mat1temp4500', 'stanford-bunny_mat1temp6500', 'stanford-bunny_mat2temp2500', 'stanford-bunny_mat2temp4500', 'stanford-bunny_mat2temp6500', 'suzanne_mat0temp2500', 'suzanne_mat0temp4500', 'suzanne_mat0temp6500', 'suzanne_mat1temp2500', 'suzanne_mat1temp4500', 'suzanne_mat1temp6500', 'suzanne_mat2temp2500', 'suzanne_mat2temp4500', 'suzanne_mat2temp6500',  'teapot_mat0temp4500', 'teapot_mat0temp6500', 'teapot_mat1temp2500', 'teapot_mat1temp4500', 'teapot_mat1temp6500', 'teapot_mat2temp2500', 'teapot_mat2temp4500', 'teapot_mat2temp6500']
  # if concepts_key == 'new_all_al_change':
  #   num_imgs = 100
  #   concepts = ['Cube_happy_stanford-bunnytex_light0temp2500', 'spottex_light2temp6500', 'Cube_happy_stanford-bunnylight2temp4500', 'stanford-bunnytex_light1temp2500', 'nefertiti_spot_suzannetex_light1temp6500', 'Sphere_nefertiti_max-plancklight2temp4500', 'Sphere_nefertiti_max-plancktex_light0temp2500', 'nefertiti_spot_suzannetex_light3temp2500', 'nefertiti_spot_suzannetex_light1temp2500', 'nefertiti_spot_suzannelight1temp4500', 'happy_homer_max-plancklight2temp4500', 'Sphere_nefertiti_max-plancklight3temp4500', 'Cube_happy_stanford-bunnylight3temp2500', 'nefertiti_spot_suzannelight2temp6500', 'teapottex_light2temp4500', 'nefertiti_spot_suzannetex_light2temp6500', 'Sphere_nefertiti_max-plancklight2temp6500', 'Sphere_nefertiti_max-plancklight1temp6500', 'stanford-bunnytex_light2temp2500', 'teapottex_light3temp4500', 'Cubetex_light0temp4500', 'Cubetex_light2temp4500', 'spot_stanford-bunny_suzannelight3temp4500', 'teapottex_light0temp4500', 'Spheretex_light2temp2500', 'happy_homer_max-plancklight1temp4500', 'nefertiti_spot_suzannelight0temp2500', 'Cube_happy_stanford-bunnytex_light1temp4500', 'teapottex_light2temp2500', 'Spheretex_light3temp6500', 'nefertiti_spot_suzannelight3temp2500', 'spot_stanford-bunny_suzannetex_light2temp2500', 'nefertiti_spot_suzannelight2temp2500', 'Cubetex_light3temp2500', 'spottex_light0temp2500', 'happy_homer_max-plancktex_light1temp6500', 'spottex_light1temp2500', 'nefertiti_spot_suzannelight0temp4500', 'spottex_light3temp2500', 'teapottex_light0temp2500', 'happy_homer_max-plancktex_light0temp4500', 'Spheretex_light1temp4500', 'Sphere_nefertiti_max-plancklight0temp4500', 'happy_homer_max-plancklight2temp6500', 'teapottex_light1temp6500', 'spot_stanford-bunny_suzannelight3temp6500', 'Cube_happy_stanford-bunnylight3temp6500', 'happy_homer_max-plancklight2temp2500', 'stanford-bunnytex_light0temp6500', 'Spheretex_light0temp4500', 'spottex_light1temp4500', 'happy_homer_max-plancktex_light0temp6500', 'spottex_light0temp4500', 'spot_stanford-bunny_suzannetex_light0temp4500', 'Cube_happy_stanford-bunnylight0temp4500', 'spot_stanford-bunny_suzannelight0temp2500', 'Cube_happy_stanford-bunnytex_light0temp6500', 'happy_homer_max-plancktex_light0temp2500', 'Spheretex_light0temp6500', 'nefertiti_spot_suzannelight3temp4500', 'nefertiti_spot_suzannetex_light2temp4500', 'spot_stanford-bunny_suzannetex_light2temp4500', 'stanford-bunnytex_light1temp6500', 'Sphere_nefertiti_max-plancklight3temp6500', 'Sphere_nefertiti_max-plancklight1temp4500', 'Sphere_nefertiti_max-plancktex_light1temp6500', 'spot_stanford-bunny_suzannetex_light1temp2500', 'teapottex_light2temp6500', 'Spheretex_light3temp2500', 'happy_homer_max-plancklight0temp4500', 'spot_stanford-bunny_suzannetex_light0temp2500', 'nefertiti_spot_suzannetex_light2temp2500', 'Sphere_nefertiti_max-plancktex_light1temp2500', 'Cubetex_light1temp2500', 'happy_homer_max-plancktex_light1temp4500', 'Cubetex_light2temp2500', 'Sphere_nefertiti_max-plancktex_light3temp2500', 'Sphere_nefertiti_max-plancktex_light2temp6500', 'happy_homer_max-plancktex_light3temp4500', 'happy_homer_max-plancklight3temp6500', 'spot_stanford-bunny_suzannelight2temp2500', 'stanford-bunnytex_light3temp2500', 'spot_stanford-bunny_suzannelight1temp2500', 'happy_homer_max-plancktex_light2temp6500', 'Cubetex_light1temp4500', 'Spheretex_light2temp4500', 'spottex_light3temp6500', 'Sphere_nefertiti_max-plancktex_light3temp4500', 'Spheretex_light1temp6500', 'spot_stanford-bunny_suzannelight2temp6500', 'Cube_happy_stanford-bunnytex_light2temp6500', 'Sphere_nefertiti_max-plancktex_light0temp6500', 'Cube_happy_stanford-bunnytex_light1temp2500', 'happy_homer_max-plancklight3temp2500', 'Cubetex_light2temp6500', 'spottex_light0temp6500', 'Sphere_nefertiti_max-plancktex_light2temp2500', 'happy_homer_max-plancklight1temp2500', 'happy_homer_max-plancktex_light1temp2500', 'Sphere_nefertiti_max-plancklight3temp2500', 'Cube_happy_stanford-bunnytex_light0temp4500', 'teapottex_light3temp2500', 'nefertiti_spot_suzannetex_light0temp4500', 'Cubetex_light3temp6500', 'happy_homer_max-plancktex_light3temp6500', 'stanford-bunnytex_light0temp4500', 'stanford-bunnytex_light3temp4500', 'spottex_light1temp6500', 'Cube_happy_stanford-bunnylight0temp6500', 'nefertiti_spot_suzannelight1temp6500', 'Cube_happy_stanford-bunnytex_light3temp4500', 'Cube_happy_stanford-bunnytex_light3temp2500', 'happy_homer_max-plancktex_light2temp4500', 'Sphere_nefertiti_max-plancktex_light1temp4500', 'spot_stanford-bunny_suzannelight1temp6500', 'Sphere_nefertiti_max-plancktex_light2temp4500', 'happy_homer_max-plancklight3temp4500', 'Spheretex_light1temp2500', 'teapottex_light0temp6500', 'Sphere_nefertiti_max-plancktex_light0temp4500', 'stanford-bunnytex_light1temp4500', 'Cube_happy_stanford-bunnylight3temp4500', 'happy_homer_max-plancktex_light3temp2500', 'nefertiti_spot_suzannetex_light0temp6500', 'Cube_happy_stanford-bunnylight1temp6500', 'nefertiti_spot_suzannetex_light0temp2500', 'happy_homer_max-plancklight0temp2500', 'Spheretex_light0temp2500', 'Sphere_nefertiti_max-plancklight2temp2500', 'nefertiti_spot_suzannelight3temp6500', 'spot_stanford-bunny_suzannetex_light3temp6500', 'spot_stanford-bunny_suzannetex_light1temp6500', 'Cube_happy_stanford-bunnytex_light3temp6500', 'spottex_light2temp2500', 'Cube_happy_stanford-bunnylight1temp2500', 'spot_stanford-bunny_suzannelight2temp4500', 'Spheretex_light2temp6500', 'nefertiti_spot_suzannelight2temp4500', 'Cubetex_light1temp6500', 'teapottex_light3temp6500', 'spottex_light3temp4500', 'Sphere_nefertiti_max-plancklight0temp2500', 'Sphere_nefertiti_max-plancktex_light3temp6500', 'teapottex_light1temp2500', 'nefertiti_spot_suzannetex_light3temp4500', 'Cube_happy_stanford-bunnylight2temp2500', 'happy_homer_max-plancklight0temp6500', 'nefertiti_spot_suzannelight1temp2500', 'Sphere_nefertiti_max-plancklight0temp6500', 'spot_stanford-bunny_suzannelight0temp4500', 'spot_stanford-bunny_suzannelight1temp4500', 'spot_stanford-bunny_suzannetex_light1temp4500', 'Spheretex_light3temp4500', 'spot_stanford-bunny_suzannelight3temp2500', 'nefertiti_spot_suzannetex_light3temp6500', 'happy_homer_max-plancktex_light2temp2500', 'spot_stanford-bunny_suzannetex_light3temp4500', 'Cubetex_light3temp4500', 'spot_stanford-bunny_suzannetex_light2temp6500', 'teapottex_light1temp4500', 'stanford-bunnytex_light2temp4500', 'happy_homer_max-plancklight1temp6500', 'Cubetex_light0temp6500', 'Cube_happy_stanford-bunnytex_light1temp6500', 'spot_stanford-bunny_suzannetex_light0temp6500', 'stanford-bunnytex_light3temp6500', 'Sphere_nefertiti_max-plancklight1temp2500', 'nefertiti_spot_suzannelight0temp6500', 'stanford-bunnytex_light2temp6500', 'spottex_light2temp4500', 'Cube_happy_stanford-bunnylight1temp4500', 'Cubetex_light0temp2500', 'stanford-bunnytex_light0temp2500', 'nefertiti_spot_suzannetex_light1temp4500', 'spot_stanford-bunny_suzannetex_light3temp2500', 'Cube_happy_stanford-bunnylight2temp6500', 'Cube_happy_stanford-bunnytex_light2temp2500', 'Cube_happy_stanford-bunnytex_light2temp4500', 'Cube_happy_stanford-bunnylight0temp2500', 'spot_stanford-bunny_suzannelight0temp6500']
  # elif concepts_key == 'new_all_albedo_change':
  #     num_imgs = 100
  #     concepts = ['Cube_happy_stanford-bunnylight0temp2500', 'Cube_happy_stanford-bunnylight0temp4500', 'Cube_happy_stanford-bunnylight0temp6500', 'Cube_happy_stanford-bunnylight1temp2500', 'Cube_happy_stanford-bunnylight1temp4500', 'Cube_happy_stanford-bunnylight1temp6500', 'Cube_happy_stanford-bunnylight2temp2500', 'Cube_happy_stanford-bunnylight2temp4500', 'Cube_happy_stanford-bunnylight2temp6500', 'Cube_happy_stanford-bunnylight3temp2500', 'Cube_happy_stanford-bunnylight3temp4500', 'Cube_happy_stanford-bunnylight3temp6500', 'Cubetex_light0temp2500', 'Cubetex_light0temp4500', 'Cubetex_light0temp6500', 'Cubetex_light1temp2500', 'Cubetex_light1temp4500', 'Cubetex_light1temp6500', 'Cubetex_light2temp2500', 'Cubetex_light2temp4500', 'Cubetex_light2temp6500', 'Cubetex_light3temp2500', 'Cubetex_light3temp4500', 'Cubetex_light3temp6500', 'Sphere_nefertiti_max-plancklight0temp2500', 'Sphere_nefertiti_max-plancklight0temp4500', 'Sphere_nefertiti_max-plancklight0temp6500', 'Sphere_nefertiti_max-plancklight1temp2500', 'Sphere_nefertiti_max-plancklight1temp4500', 'Sphere_nefertiti_max-plancklight1temp6500', 'Sphere_nefertiti_max-plancklight2temp2500', 'Sphere_nefertiti_max-plancklight2temp4500', 'Sphere_nefertiti_max-plancklight2temp6500', 'Sphere_nefertiti_max-plancklight3temp2500', 'Sphere_nefertiti_max-plancklight3temp4500', 'Sphere_nefertiti_max-plancklight3temp6500', 'Spheretex_light0temp2500', 'Spheretex_light0temp4500', 'Spheretex_light0temp6500', 'Spheretex_light1temp2500', 'Spheretex_light1temp4500', 'Spheretex_light1temp6500', 'Spheretex_light2temp2500', 'Spheretex_light2temp4500', 'Spheretex_light2temp6500', 'Spheretex_light3temp2500', 'Spheretex_light3temp4500', 'Spheretex_light3temp6500', 'happy_homer_max-plancklight0temp2500', 'happy_homer_max-plancklight0temp4500', 'happy_homer_max-plancklight0temp6500', 'happy_homer_max-plancklight1temp2500', 'happy_homer_max-plancklight1temp4500', 'happy_homer_max-plancklight1temp6500', 'happy_homer_max-plancklight2temp2500', 'happy_homer_max-plancklight2temp4500', 'happy_homer_max-plancklight2temp6500', 'happy_homer_max-plancklight3temp2500', 'happy_homer_max-plancklight3temp4500', 'happy_homer_max-plancklight3temp6500', 'nefertiti_spot_suzannelight0temp2500', 'nefertiti_spot_suzannelight0temp4500', 'nefertiti_spot_suzannelight0temp6500', 'nefertiti_spot_suzannelight1temp2500', 'nefertiti_spot_suzannelight1temp4500', 'nefertiti_spot_suzannelight1temp6500', 'nefertiti_spot_suzannelight2temp2500', 'nefertiti_spot_suzannelight2temp4500', 'nefertiti_spot_suzannelight2temp6500', 'nefertiti_spot_suzannelight3temp2500', 'nefertiti_spot_suzannelight3temp4500', 'nefertiti_spot_suzannelight3temp6500', 'spot_stanford-bunny_suzannelight0temp2500', 'spot_stanford-bunny_suzannelight0temp4500', 'spot_stanford-bunny_suzannelight0temp6500', 'spot_stanford-bunny_suzannelight1temp2500', 'spot_stanford-bunny_suzannelight1temp4500', 'spot_stanford-bunny_suzannelight1temp6500', 'spot_stanford-bunny_suzannelight2temp2500', 'spot_stanford-bunny_suzannelight2temp4500', 'spot_stanford-bunny_suzannelight2temp6500', 'spot_stanford-bunny_suzannelight3temp2500', 'spot_stanford-bunny_suzannelight3temp4500', 'spot_stanford-bunny_suzannelight3temp6500', 'spottex_light0temp2500', 'spottex_light0temp4500', 'spottex_light0temp6500', 'spottex_light1temp2500', 'spottex_light1temp4500', 'spottex_light1temp6500', 'spottex_light2temp2500', 'spottex_light2temp4500', 'spottex_light2temp6500', 'spottex_light3temp2500', 'spottex_light3temp4500', 'spottex_light3temp6500', 'stanford-bunnytex_light0temp2500', 'stanford-bunnytex_light0temp4500', 'stanford-bunnytex_light0temp6500', 'stanford-bunnytex_light1temp2500', 'stanford-bunnytex_light1temp4500', 'stanford-bunnytex_light1temp6500', 'stanford-bunnytex_light2temp2500', 'stanford-bunnytex_light2temp4500', 'stanford-bunnytex_light2temp6500', 'stanford-bunnytex_light3temp2500', 'stanford-bunnytex_light3temp4500', 'stanford-bunnytex_light3temp6500', 'teapottex_light0temp2500', 'teapottex_light0temp4500', 'teapottex_light0temp6500', 'teapottex_light1temp2500', 'teapottex_light1temp4500', 'teapottex_light1temp6500', 'teapottex_light2temp2500', 'teapottex_light2temp4500', 'teapottex_light2temp6500', 'teapottex_light3temp2500', 'teapottex_light3temp4500', 'teapottex_light3temp6500', 'Cube_happy_stanford-bunnytex_light0temp2500', 'Cube_happy_stanford-bunnytex_light0temp4500', 'Cube_happy_stanford-bunnytex_light0temp6500', 'Cube_happy_stanford-bunnytex_light1temp2500', 'Cube_happy_stanford-bunnytex_light1temp4500', 'Cube_happy_stanford-bunnytex_light1temp6500', 'Cube_happy_stanford-bunnytex_light2temp2500', 'Cube_happy_stanford-bunnytex_light2temp4500', 'Cube_happy_stanford-bunnytex_light2temp6500', 'Cube_happy_stanford-bunnytex_light3temp2500', 'Cube_happy_stanford-bunnytex_light3temp4500', 'Cube_happy_stanford-bunnytex_light3temp6500', 'Sphere_nefertiti_max-plancktex_light0temp2500', 'Sphere_nefertiti_max-plancktex_light0temp4500', 'Sphere_nefertiti_max-plancktex_light0temp6500', 'Sphere_nefertiti_max-plancktex_light1temp2500', 'Sphere_nefertiti_max-plancktex_light1temp4500', 'Sphere_nefertiti_max-plancktex_light1temp6500', 'Sphere_nefertiti_max-plancktex_light2temp2500', 'Sphere_nefertiti_max-plancktex_light2temp4500', 'Sphere_nefertiti_max-plancktex_light2temp6500', 'Sphere_nefertiti_max-plancktex_light3temp2500', 'Sphere_nefertiti_max-plancktex_light3temp4500', 'Sphere_nefertiti_max-plancktex_light3temp6500', 'happy_homer_max-plancktex_light0temp2500', 'happy_homer_max-plancktex_light0temp4500', 'happy_homer_max-plancktex_light0temp6500', 'happy_homer_max-plancktex_light1temp2500', 'happy_homer_max-plancktex_light1temp4500', 'happy_homer_max-plancktex_light1temp6500', 'happy_homer_max-plancktex_light2temp2500', 'happy_homer_max-plancktex_light2temp4500', 'happy_homer_max-plancktex_light2temp6500', 'happy_homer_max-plancktex_light3temp2500', 'happy_homer_max-plancktex_light3temp4500', 'happy_homer_max-plancktex_light3temp6500', 'nefertiti_spot_suzannetex_light0temp2500', 'nefertiti_spot_suzannetex_light0temp4500', 'nefertiti_spot_suzannetex_light0temp6500', 'nefertiti_spot_suzannetex_light1temp2500', 'nefertiti_spot_suzannetex_light1temp4500', 'nefertiti_spot_suzannetex_light1temp6500', 'nefertiti_spot_suzannetex_light2temp2500', 'nefertiti_spot_suzannetex_light2temp4500', 'nefertiti_spot_suzannetex_light2temp6500', 'nefertiti_spot_suzannetex_light3temp2500', 'nefertiti_spot_suzannetex_light3temp4500', 'nefertiti_spot_suzannetex_light3temp6500', 'spot_stanford-bunny_suzannetex_light0temp2500', 'spot_stanford-bunny_suzannetex_light0temp4500', 'spot_stanford-bunny_suzannetex_light0temp6500', 'spot_stanford-bunny_suzannetex_light1temp2500', 'spot_stanford-bunny_suzannetex_light1temp4500', 'spot_stanford-bunny_suzannetex_light1temp6500', 'spot_stanford-bunny_suzannetex_light2temp2500', 'spot_stanford-bunny_suzannetex_light2temp4500', 'spot_stanford-bunny_suzannetex_light2temp6500', 'spot_stanford-bunny_suzannetex_light3temp2500', 'spot_stanford-bunny_suzannetex_light3temp4500', 'spot_stanford-bunny_suzannetex_light3temp6500']

  # elif concepts_key =='new_all_ill_change':
  #     num_imgs = 43
  #     concepts = ['Cube_happy_stanford-bunny_mat0temp2500', 'Cube_happy_stanford-bunny_mat0temp4500', 'Cube_happy_stanford-bunny_mat0temp6500', 'Cube_happy_stanford-bunny_mat1temp2500', 'Cube_happy_stanford-bunny_mat1temp4500', 'Cube_happy_stanford-bunny_mat1temp6500', 'Cube_happy_stanford-bunny_mat2temp2500', 'Cube_happy_stanford-bunny_mat2temp4500', 'Cube_happy_stanford-bunny_mat2temp6500', 'Cube_tex_mat0temp2500', 'Cube_tex_mat0temp4500', 'Cube_tex_mat0temp6500', 'Cube_tex_mat1temp2500', 'Cube_tex_mat1temp4500', 'Cube_tex_mat1temp6500', 'Cube_tex_mat2temp2500', 'Cube_tex_mat2temp4500', 'Cube_tex_mat2temp6500', 'Sphere_nefertiti_max-planck_mat0temp2500', 'Sphere_nefertiti_max-planck_mat0temp4500', 'Sphere_nefertiti_max-planck_mat0temp6500', 'Sphere_nefertiti_max-planck_mat1temp2500', 'Sphere_nefertiti_max-planck_mat1temp4500', 'Sphere_nefertiti_max-planck_mat1temp6500', 'Sphere_nefertiti_max-planck_mat2temp2500', 'Sphere_nefertiti_max-planck_mat2temp4500', 'Sphere_nefertiti_max-planck_mat2temp6500', 'Sphere_tex_mat0temp2500', 'Sphere_tex_mat0temp4500', 'Sphere_tex_mat0temp6500', 'Sphere_tex_mat1temp2500', 'Sphere_tex_mat1temp4500', 'Sphere_tex_mat1temp6500', 'Sphere_tex_mat2temp2500', 'Sphere_tex_mat2temp4500', 'Sphere_tex_mat2temp6500', 'happy_homer_max-planck_mat0temp2500', 'happy_homer_max-planck_mat0temp4500', 'happy_homer_max-planck_mat0temp6500', 'happy_homer_max-planck_mat1temp2500', 'happy_homer_max-planck_mat1temp4500', 'happy_homer_max-planck_mat1temp6500', 'happy_homer_max-planck_mat2temp2500', 'happy_homer_max-planck_mat2temp4500', 'happy_homer_max-planck_mat2temp6500', 'nefertiti_spot_suzanne_mat0temp2500', 'nefertiti_spot_suzanne_mat0temp4500', 'nefertiti_spot_suzanne_mat0temp6500', 'nefertiti_spot_suzanne_mat1temp2500', 'nefertiti_spot_suzanne_mat1temp4500', 'nefertiti_spot_suzanne_mat1temp6500', 'nefertiti_spot_suzanne_mat2temp2500', 'nefertiti_spot_suzanne_mat2temp4500', 'nefertiti_spot_suzanne_mat2temp6500', 'spot_stanford-bunny_suzanne_mat0temp2500', 'spot_stanford-bunny_suzanne_mat0temp4500', 'spot_stanford-bunny_suzanne_mat0temp6500', 'spot_stanford-bunny_suzanne_mat1temp2500', 'spot_stanford-bunny_suzanne_mat1temp4500', 'spot_stanford-bunny_suzanne_mat1temp6500', 'spot_stanford-bunny_suzanne_mat2temp2500', 'spot_stanford-bunny_suzanne_mat2temp4500', 'spot_stanford-bunny_suzanne_mat2temp6500', 'spot_tex_mat0temp2500', 'spot_tex_mat0temp4500', 'spot_tex_mat0temp6500', 'spot_tex_mat1temp2500', 'spot_tex_mat1temp4500', 'spot_tex_mat1temp6500', 'spot_tex_mat2temp2500', 'spot_tex_mat2temp4500', 'spot_tex_mat2temp6500', 'stanford-bunny_tex_mat0temp2500', 'stanford-bunny_tex_mat0temp4500', 'stanford-bunny_tex_mat0temp6500', 'stanford-bunny_tex_mat1temp2500', 'stanford-bunny_tex_mat1temp4500', 'stanford-bunny_tex_mat1temp6500', 'stanford-bunny_tex_mat2temp2500', 'stanford-bunny_tex_mat2temp4500', 'stanford-bunny_tex_mat2temp6500', 'teapot_tex_mat0temp2500', 'teapot_tex_mat0temp4500', 'teapot_tex_mat0temp6500', 'teapot_tex_mat1temp2500', 'teapot_tex_mat1temp4500', 'teapot_tex_mat1temp6500', 'teapot_tex_mat2temp2500', 'teapot_tex_mat2temp4500', 'teapot_tex_mat2temp6500', 'Cube_happy_stanford-bunny_tex_mat0temp2500', 'Cube_happy_stanford-bunny_tex_mat0temp4500', 'Cube_happy_stanford-bunny_tex_mat0temp6500', 'Cube_happy_stanford-bunny_tex_mat1temp2500', 'Cube_happy_stanford-bunny_tex_mat1temp4500', 'Cube_happy_stanford-bunny_tex_mat1temp6500', 'Cube_happy_stanford-bunny_tex_mat2temp2500', 'Cube_happy_stanford-bunny_tex_mat2temp4500', 'Cube_happy_stanford-bunny_tex_mat2temp6500', 'Sphere_nefertiti_max-planck_tex_mat0temp2500', 'Sphere_nefertiti_max-planck_tex_mat0temp4500', 'Sphere_nefertiti_max-planck_tex_mat0temp6500', 'Sphere_nefertiti_max-planck_tex_mat1temp2500', 'Sphere_nefertiti_max-planck_tex_mat1temp4500', 'Sphere_nefertiti_max-planck_tex_mat1temp6500', 'Sphere_nefertiti_max-planck_tex_mat2temp2500', 'Sphere_nefertiti_max-planck_tex_mat2temp4500', 'Sphere_nefertiti_max-planck_tex_mat2temp6500', 'happy_homer_max-planck_tex_mat0temp2500', 'happy_homer_max-planck_tex_mat0temp4500', 'happy_homer_max-planck_tex_mat0temp6500', 'happy_homer_max-planck_tex_mat1temp2500', 'happy_homer_max-planck_tex_mat1temp4500', 'happy_homer_max-planck_tex_mat1temp6500', 'happy_homer_max-planck_tex_mat2temp2500', 'happy_homer_max-planck_tex_mat2temp4500', 'happy_homer_max-planck_tex_mat2temp6500', 'nefertiti_spot_suzanne_tex_mat0temp2500', 'nefertiti_spot_suzanne_tex_mat0temp4500', 'nefertiti_spot_suzanne_tex_mat0temp6500', 'nefertiti_spot_suzanne_tex_mat1temp2500', 'nefertiti_spot_suzanne_tex_mat1temp4500', 'nefertiti_spot_suzanne_tex_mat1temp6500', 'nefertiti_spot_suzanne_tex_mat2temp2500', 'nefertiti_spot_suzanne_tex_mat2temp4500', 'nefertiti_spot_suzanne_tex_mat2temp6500', 'spot_stanford-bunny_suzanne_tex_mat0temp2500', 'spot_stanford-bunny_suzanne_tex_mat0temp4500', 'spot_stanford-bunny_suzanne_tex_mat0temp6500', 'spot_stanford-bunny_suzanne_tex_mat1temp2500', 'spot_stanford-bunny_suzanne_tex_mat1temp4500', 'spot_stanford-bunny_suzanne_tex_mat1temp6500', 'spot_stanford-bunny_suzanne_tex_mat2temp2500', 'spot_stanford-bunny_suzanne_tex_mat2temp4500', 'spot_stanford-bunny_suzanne_tex_mat2temp6500']

  # elif concepts_key == 'new_all_albedo_change_addi':
  #     concepts = ['Sphere_Cube_teapotlight0temp2500', 'Sphere_Cube_teapotlight0temp4500', 'Sphere_Cube_teapotlight0temp6500', 'Sphere_Cube_teapotlight1temp2500', 'Sphere_Cube_teapotlight1temp4500', 'Sphere_Cube_teapotlight1temp6500', 'Sphere_Cube_teapotlight2temp2500', 'Sphere_Cube_teapotlight2temp4500', 'Sphere_Cube_teapotlight2temp6500', 'Sphere_Cube_teapotlight3temp2500', 'Sphere_Cube_teapotlight3temp4500', 'Sphere_Cube_teapotlight3temp6500', 'Sphere_Cube_teapottex_light0temp2500', 'Sphere_Cube_teapottex_light0temp4500', 'Sphere_Cube_teapottex_light0temp6500', 'Sphere_Cube_teapottex_light1temp2500', 'Sphere_Cube_teapottex_light1temp4500', 'Sphere_Cube_teapottex_light1temp6500', 'Sphere_Cube_teapottex_light2temp2500', 'Sphere_Cube_teapottex_light2temp4500', 'Sphere_Cube_teapottex_light2temp6500', 'Sphere_Cube_teapottex_light3temp2500', 'Sphere_Cube_teapottex_light3temp4500', 'Sphere_Cube_teapottex_light3temp6500', 'happy_homer_Spherelight0temp2500', 'happy_homer_Spherelight0temp4500', 'happy_homer_Spherelight0temp6500', 'happy_homer_Spherelight1temp2500', 'happy_homer_Spherelight1temp4500', 'happy_homer_Spherelight1temp6500', 'happy_homer_Spherelight2temp2500', 'happy_homer_Spherelight2temp4500', 'happy_homer_Spherelight2temp6500', 'happy_homer_Spherelight3temp2500', 'happy_homer_Spherelight3temp4500', 'happy_homer_Spherelight3temp6500', 'happy_homer_Spheretex_light0temp2500', 'happy_homer_Spheretex_light0temp4500', 'happy_homer_Spheretex_light0temp6500', 'happy_homer_Spheretex_light1temp2500', 'happy_homer_Spheretex_light1temp4500', 'happy_homer_Spheretex_light1temp6500', 'happy_homer_Spheretex_light2temp2500', 'happy_homer_Spheretex_light2temp4500', 'happy_homer_Spheretex_light2temp6500', 'happy_homer_Spheretex_light3temp2500', 'happy_homer_Spheretex_light3temp4500', 'happy_homer_Spheretex_light3temp6500', 'happy_max-planck_spotlight0temp2500', 'happy_max-planck_spotlight0temp4500', 'happy_max-planck_spotlight0temp6500', 'happy_max-planck_spotlight1temp2500', 'happy_max-planck_spotlight1temp4500', 'happy_max-planck_spotlight1temp6500', 'happy_max-planck_spotlight2temp2500', 'happy_max-planck_spotlight2temp4500', 'happy_max-planck_spotlight2temp6500', 'happy_max-planck_spotlight3temp2500', 'happy_max-planck_spotlight3temp4500', 'happy_max-planck_spotlight3temp6500', 'happy_max-planck_spottex_light0temp2500', 'happy_max-planck_spottex_light0temp4500', 'happy_max-planck_spottex_light0temp6500', 'happy_max-planck_spottex_light1temp2500', 'happy_max-planck_spottex_light1temp4500', 'happy_max-planck_spottex_light1temp6500', 'happy_max-planck_spottex_light2temp2500', 'happy_max-planck_spottex_light2temp4500', 'happy_max-planck_spottex_light2temp6500', 'happy_max-planck_spottex_light3temp2500', 'happy_max-planck_spottex_light3temp4500', 'happy_max-planck_spottex_light3temp6500', 'happytex_light0temp2500', 'happytex_light0temp4500', 'happytex_light0temp6500', 'happytex_light1temp2500', 'happytex_light1temp4500', 'happytex_light1temp6500', 'happytex_light2temp2500', 'happytex_light2temp4500', 'happytex_light2temp6500', 'happytex_light3temp2500', 'happytex_light3temp4500', 'happytex_light3temp6500', 'homer_max-planck_suzannelight0temp2500', 'homer_max-planck_suzannelight0temp4500', 'homer_max-planck_suzannelight0temp6500', 'homer_max-planck_suzannelight1temp2500', 'homer_max-planck_suzannelight1temp4500', 'homer_max-planck_suzannelight1temp6500', 'homer_max-planck_suzannelight2temp2500', 'homer_max-planck_suzannelight2temp4500', 'homer_max-planck_suzannelight2temp6500', 'homer_max-planck_suzannelight3temp2500', 'homer_max-planck_suzannelight3temp4500', 'homer_max-planck_suzannelight3temp6500', 'homer_max-planck_suzannetex_light0temp2500', 'homer_max-planck_suzannetex_light0temp4500', 'homer_max-planck_suzannetex_light0temp6500', 'homer_max-planck_suzannetex_light1temp2500', 'homer_max-planck_suzannetex_light1temp4500', 'homer_max-planck_suzannetex_light1temp6500', 'homer_max-planck_suzannetex_light2temp2500', 'homer_max-planck_suzannetex_light2temp4500', 'homer_max-planck_suzannetex_light2temp6500', 'homer_max-planck_suzannetex_light3temp2500', 'homer_max-planck_suzannetex_light3temp4500', 'homer_max-planck_suzannetex_light3temp6500', 'homertex_light0temp2500', 'homertex_light0temp4500', 'homertex_light0temp6500', 'homertex_light1temp2500', 'homertex_light1temp4500', 'homertex_light1temp6500', 'homertex_light2temp2500', 'homertex_light2temp4500', 'homertex_light2temp6500', 'homertex_light3temp2500', 'homertex_light3temp4500', 'homertex_light3temp6500', 'max-planck_stanford-bunny_teapotlight0temp2500', 'max-planck_stanford-bunny_teapotlight0temp4500', 'max-planck_stanford-bunny_teapotlight0temp6500', 'max-planck_stanford-bunny_teapotlight1temp2500', 'max-planck_stanford-bunny_teapotlight1temp4500', 'max-planck_stanford-bunny_teapotlight1temp6500', 'max-planck_stanford-bunny_teapotlight2temp2500', 'max-planck_stanford-bunny_teapotlight2temp4500', 'max-planck_stanford-bunny_teapotlight2temp6500', 'max-planck_stanford-bunny_teapotlight3temp2500', 'max-planck_stanford-bunny_teapotlight3temp4500', 'max-planck_stanford-bunny_teapotlight3temp6500', 'max-planck_stanford-bunny_teapottex_light0temp2500', 'max-planck_stanford-bunny_teapottex_light0temp4500', 'max-planck_stanford-bunny_teapottex_light0temp6500', 'max-planck_stanford-bunny_teapottex_light1temp2500', 'max-planck_stanford-bunny_teapottex_light1temp4500', 'max-planck_stanford-bunny_teapottex_light1temp6500', 'max-planck_stanford-bunny_teapottex_light2temp2500', 'max-planck_stanford-bunny_teapottex_light2temp4500', 'max-planck_stanford-bunny_teapottex_light2temp6500', 'max-planck_stanford-bunny_teapottex_light3temp2500', 'max-planck_stanford-bunny_teapottex_light3temp4500', 'max-planck_stanford-bunny_teapottex_light3temp6500', 'max-plancktex_light0temp2500', 'max-plancktex_light0temp4500', 'max-plancktex_light0temp6500', 'max-plancktex_light1temp2500', 'max-plancktex_light1temp4500', 'max-plancktex_light1temp6500', 'max-plancktex_light2temp2500', 'max-plancktex_light2temp4500', 'max-plancktex_light2temp6500', 'max-plancktex_light3temp2500', 'max-plancktex_light3temp4500', 'max-plancktex_light3temp6500', 'nefertititex_light0temp2500', 'nefertititex_light0temp4500', 'nefertititex_light0temp6500', 'nefertititex_light1temp2500', 'nefertititex_light1temp4500', 'nefertititex_light1temp6500', 'nefertititex_light2temp2500', 'nefertititex_light2temp4500', 'nefertititex_light2temp6500', 'nefertititex_light3temp2500', 'nefertititex_light3temp4500', 'nefertititex_light3temp6500', 'suzannetex_light0temp2500', 'suzannetex_light0temp4500', 'suzannetex_light0temp6500', 'suzannetex_light1temp2500', 'suzannetex_light1temp4500', 'suzannetex_light1temp6500', 'suzannetex_light2temp2500', 'suzannetex_light2temp4500', 'suzannetex_light2temp6500', 'suzannetex_light3temp2500', 'suzannetex_light3temp4500', 'suzannetex_light3temp6500']

  # elif concepts_key =='new_all_ill_change_addi':
  #     num_imgs = 43
  #     concepts = ['Sphere_Cube_teapot_mat0temp2500', 'Sphere_Cube_teapot_mat0temp4500', 'Sphere_Cube_teapot_mat0temp6500', 'Sphere_Cube_teapot_mat1temp2500', 'Sphere_Cube_teapot_mat1temp4500', 'Sphere_Cube_teapot_mat1temp6500', 'Sphere_Cube_teapot_mat2temp2500', 'Sphere_Cube_teapot_mat2temp4500', 'Sphere_Cube_teapot_mat2temp6500', 'Sphere_Cube_teapot_tex_mat0temp2500', 'Sphere_Cube_teapot_tex_mat0temp4500', 'Sphere_Cube_teapot_tex_mat0temp6500', 'Sphere_Cube_teapot_tex_mat1temp2500', 'Sphere_Cube_teapot_tex_mat1temp4500', 'Sphere_Cube_teapot_tex_mat1temp6500', 'Sphere_Cube_teapot_tex_mat2temp2500', 'Sphere_Cube_teapot_tex_mat2temp4500', 'Sphere_Cube_teapot_tex_mat2temp6500', 'happy_homer_Sphere_mat0temp2500', 'happy_homer_Sphere_mat0temp4500', 'happy_homer_Sphere_mat0temp6500', 'happy_homer_Sphere_mat1temp2500', 'happy_homer_Sphere_mat1temp4500', 'happy_homer_Sphere_mat1temp6500', 'happy_homer_Sphere_mat2temp2500', 'happy_homer_Sphere_mat2temp4500', 'happy_homer_Sphere_mat2temp6500', 'happy_homer_Sphere_tex_mat0temp2500', 'happy_homer_Sphere_tex_mat0temp4500', 'happy_homer_Sphere_tex_mat0temp6500', 'happy_homer_Sphere_tex_mat1temp2500', 'happy_homer_Sphere_tex_mat1temp4500', 'happy_homer_Sphere_tex_mat1temp6500', 'happy_homer_Sphere_tex_mat2temp2500', 'happy_homer_Sphere_tex_mat2temp4500', 'happy_homer_Sphere_tex_mat2temp6500', 'happy_max-planck_spot_mat0temp2500', 'happy_max-planck_spot_mat0temp4500', 'happy_max-planck_spot_mat0temp6500', 'happy_max-planck_spot_mat1temp2500', 'happy_max-planck_spot_mat1temp4500', 'happy_max-planck_spot_mat1temp6500', 'happy_max-planck_spot_mat2temp2500', 'happy_max-planck_spot_mat2temp4500', 'happy_max-planck_spot_mat2temp6500', 'happy_max-planck_spot_tex_mat0temp2500', 'happy_max-planck_spot_tex_mat0temp4500', 'happy_max-planck_spot_tex_mat0temp6500', 'happy_max-planck_spot_tex_mat1temp2500', 'happy_max-planck_spot_tex_mat1temp4500', 'happy_max-planck_spot_tex_mat1temp6500', 'happy_max-planck_spot_tex_mat2temp2500', 'happy_max-planck_spot_tex_mat2temp4500', 'happy_max-planck_spot_tex_mat2temp6500', 'happy_tex_mat0temp2500', 'happy_tex_mat0temp4500', 'happy_tex_mat0temp6500', 'happy_tex_mat1temp2500', 'happy_tex_mat1temp4500', 'happy_tex_mat1temp6500', 'happy_tex_mat2temp2500', 'happy_tex_mat2temp4500', 'happy_tex_mat2temp6500', 'homer_max-planck_suzanne_mat0temp2500', 'homer_max-planck_suzanne_mat0temp4500', 'homer_max-planck_suzanne_mat0temp6500', 'homer_max-planck_suzanne_mat1temp2500', 'homer_max-planck_suzanne_mat1temp4500', 'homer_max-planck_suzanne_mat1temp6500', 'homer_max-planck_suzanne_mat2temp2500', 'homer_max-planck_suzanne_mat2temp4500', 'homer_max-planck_suzanne_mat2temp6500', 'homer_max-planck_suzanne_tex_mat0temp2500', 'homer_max-planck_suzanne_tex_mat0temp4500', 'homer_max-planck_suzanne_tex_mat0temp6500', 'homer_max-planck_suzanne_tex_mat1temp2500', 'homer_max-planck_suzanne_tex_mat1temp4500', 'homer_max-planck_suzanne_tex_mat1temp6500', 'homer_max-planck_suzanne_tex_mat2temp2500', 'homer_max-planck_suzanne_tex_mat2temp4500', 'homer_max-planck_suzanne_tex_mat2temp6500', 'homer_tex_mat0temp2500', 'homer_tex_mat0temp4500', 'homer_tex_mat0temp6500', 'homer_tex_mat1temp2500', 'homer_tex_mat1temp4500', 'homer_tex_mat1temp6500', 'homer_tex_mat2temp2500', 'homer_tex_mat2temp4500', 'homer_tex_mat2temp6500', 'max-planck_stanford-bunny_teapot_mat0temp2500', 'max-planck_stanford-bunny_teapot_mat0temp4500', 'max-planck_stanford-bunny_teapot_mat0temp6500', 'max-planck_stanford-bunny_teapot_mat1temp2500', 'max-planck_stanford-bunny_teapot_mat1temp4500', 'max-planck_stanford-bunny_teapot_mat1temp6500', 'max-planck_stanford-bunny_teapot_mat2temp2500', 'max-planck_stanford-bunny_teapot_mat2temp4500', 'max-planck_stanford-bunny_teapot_mat2temp6500', 'max-planck_stanford-bunny_teapot_tex_mat0temp2500', 'max-planck_stanford-bunny_teapot_tex_mat0temp4500', 'max-planck_stanford-bunny_teapot_tex_mat0temp6500', 'max-planck_stanford-bunny_teapot_tex_mat1temp2500', 'max-planck_stanford-bunny_teapot_tex_mat1temp4500', 'max-planck_stanford-bunny_teapot_tex_mat1temp6500', 'max-planck_stanford-bunny_teapot_tex_mat2temp2500', 'max-planck_stanford-bunny_teapot_tex_mat2temp4500', 'max-planck_stanford-bunny_teapot_tex_mat2temp6500', 'max-planck_tex_mat0temp2500', 'max-planck_tex_mat0temp4500', 'max-planck_tex_mat0temp6500', 'max-planck_tex_mat1temp2500', 'max-planck_tex_mat1temp4500', 'max-planck_tex_mat1temp6500', 'max-planck_tex_mat2temp2500', 'max-planck_tex_mat2temp4500', 'max-planck_tex_mat2temp6500', 'nefertiti_tex_mat0temp2500', 'nefertiti_tex_mat0temp4500', 'nefertiti_tex_mat0temp6500', 'nefertiti_tex_mat1temp2500', 'nefertiti_tex_mat1temp4500', 'nefertiti_tex_mat1temp6500', 'nefertiti_tex_mat2temp2500', 'nefertiti_tex_mat2temp4500', 'nefertiti_tex_mat2temp6500', 'suzanne_tex_mat0temp2500', 'suzanne_tex_mat0temp4500', 'suzanne_tex_mat0temp6500', 'suzanne_tex_mat1temp2500', 'suzanne_tex_mat1temp4500', 'suzanne_tex_mat1temp6500', 'suzanne_tex_mat2temp2500', 'suzanne_tex_mat2temp4500', 'suzanne_tex_mat2temp6500']

  # elif concepts_key == 'real_multi_ill':
  #   num_imgs = 24
  #   concepts = ['everett_kitchen2', 'everett_lobby13', 'everett_lobby19', 'everett_kitchen4', 'everett_lobby1', 'everett_lobby4', 'everett_kitchen14', 'everett_lobby15', 'everett_kitchen8', 'everett_lobby12', 'everett_kitchen5', 'everett_kitchen17', 'everett_kitchen6', 'everett_kitchen18', 'everett_lobby14', 'everett_lobby6', 'everett_lobby17', 'everett_lobby2', 'everett_lobby3', 'everett_dining1', 'everett_lobby11', 'everett_living4', 'everett_kitchen9', 'everett_lobby16', 'everett_kitchen7', 'everett_lobby18', 'everett_dining2', 'everett_kitchen12', 'everett_lobby20', 'everett_living2']

  # elif concepts_key =='real_il3':
  #   num_imgs = 43
  #   concepts = ['gourd1', 'gourd2','apple']
  # return concepts, num_imgs

def plot_results_(results, random_counterpart=None, random_concepts=None, num_random_exp=100,
    min_p_val=0.05,plot=False,verbose=False, fig_name='cur_test.png'):
  """Helper function to organize results.
  When run in a notebook, outputs a matplotlib bar plot of the
  TCAV scores for all bottlenecks for each concept, replacing the
  bars with asterisks when the TCAV score is not statistically significant.
  If you ran TCAV with a random_counterpart, supply it here, otherwise supply random_concepts.
  If you get unexpected output, make sure you are using the correct keywords.

  Args:
    results: dictionary of results from TCAV runs.
    random_counterpart: name of the random_counterpart used, if it was used. 
    random_concepts: list of random experiments that were run. 
    num_random_exp: number of random experiments that were run.
    min_p_val: minimum p value for statistical significance
  """

  # helper function, returns if this is a random concept
  def is_random_concept(concept):
    if random_counterpart:
      return random_counterpart == concept
    
    elif random_concepts:
      return concept in random_concepts

    else:
      return 'random500_' in concept

  # print class, it will be the same for all
  
  # prepare data
  # dict with keys of concepts containing dict with bottlenecks
  result_summary = {}
  # random
  random_i_ups = {}
#   print("results ",results)
  for result in results:
    # print("result ",result)
    if result['cav_concept'] not in result_summary:
      result_summary[result['cav_concept']] = {}
    
    if result['bottleneck'] not in result_summary[result['cav_concept']]:
      result_summary[result['cav_concept']][result['bottleneck']] = []
    
    result_summary[result['cav_concept']][result['bottleneck']].append(result)

    # store random
    if is_random_concept(result['cav_concept']):
      if result['bottleneck'] not in random_i_ups:
        random_i_ups[result['bottleneck']] = []
        
      random_i_ups[result['bottleneck']].append(result['i_up'])
    
  # to plot, must massage data again 
  plot_data = {}
  sig_data = {}
  plot_concepts = []
    
  # print concepts and classes with indentation
  for concept in result_summary:
        
    # if not random
    if not is_random_concept(concept):
      if verbose:
          print(" ", "Concept =", concept)
      plot_concepts.append(concept)

      for bottleneck in result_summary[concept]:
        i_ups = [item['i_up'] for item in result_summary[concept][bottleneck]]
        
        # Calculate statistical significance
        # print("random_i_ups[bottleneck] ",random_i_ups, bottleneck)
        _, p_val = ttest_ind(random_i_ups[bottleneck], i_ups)
                  
        if bottleneck not in plot_data:
          plot_data[bottleneck] = {'bn_vals': [], 'bn_stds': [], 'significant': []}
          sig_data[bottleneck] = {}

        if p_val > min_p_val:
          # statistically insignificant
        #   plot_data[bottleneck]['bn_vals'].append(0.01)
        #   plot_data[bottleneck]['bn_stds'].append(0)
        #   plot_data[bottleneck]['significant'].append(False)
          plot_data[bottleneck]['bn_vals'].append(np.mean(i_ups))
          plot_data[bottleneck]['bn_stds'].append(np.std(i_ups))
          plot_data[bottleneck]['significant'].append(False)
          sig_data[bottleneck][concept] = False
            
        else:
          plot_data[bottleneck]['bn_vals'].append(np.mean(i_ups))
          plot_data[bottleneck]['bn_stds'].append(np.std(i_ups))
          plot_data[bottleneck]['significant'].append(True)
          sig_data[bottleneck][concept] = True
        
        if verbose:
            print(3 * " ", "Bottleneck =", ("%s. TCAV Score = %.2f (+- %.2f), "
                "random was %.2f (+- %.2f). p-val = %.3f (%s)") % (
                bottleneck, np.mean(i_ups), np.std(i_ups),
                np.mean(random_i_ups[bottleneck]),
                np.std(random_i_ups[bottleneck]), p_val,
                "not significant" if p_val > min_p_val else "significant"))
        
  # subtract number of random experiments
  if random_counterpart:
    num_concepts = len(result_summary) - 1
  elif random_concepts:
    num_concepts = len(result_summary) - len(random_concepts)
  else: 
    num_concepts = len(result_summary) - num_random_exp
    
  num_bottlenecks = len(plot_data)
  bar_width = 0.35
  if plot==True:
    
      # create location for each bar. scale by an appropriate factor to ensure 
      # the final plot doesn't have any parts overlapping
      index = np.arange(num_concepts) * bar_width * (num_bottlenecks + 1)

      # matplotlib
      fig, ax = plt.subplots()

      # draw all bottlenecks individually
      for i, [bn, vals] in enumerate(plot_data.items()):
        bar = ax.bar(index + i * bar_width, vals['bn_vals'],
            bar_width, yerr=vals['bn_stds'], label=bn)

        # draw stars to mark bars that are stastically insignificant to 
        # show them as different from others
        for j, significant in enumerate(vals['significant']):
          if not significant:
            ax.text(index[j] + i * bar_width - 0.1, 0.01, "*",
                fontdict = {'weight': 'bold', 'size': 16,
                'color': bar.patches[0].get_facecolor()})

      # set properties
      ax.set_title('TCAV Scores for each concept and bottleneck')
      ax.set_ylabel('TCAV Score')
      ax.set_xticks(index + num_bottlenecks * bar_width / 2)
      ax.set_xticklabels(plot_concepts,rotation=20)
      ax.legend()
      fig.tight_layout()
      plt.savefig(fig_name)
  return plot_data, sig_data

def plot(inp):
    plt.figure()
    try:
        im = inp.squeeze(0).detach().cpu().numpy()
        im = np.moveaxis(im,0,-1)
    except:
        im = inp.detach().cpu().numpy()
    plt.imshow(im)
    
def plot_gray(inp):
    plt.figure()
    try:
        im = inp.squeeze(0).detach().cpu().numpy()
    except:
        im = inp.detach().cpu().numpy()
    plt.imshow(im,cmap='gray')

def plot(inp):
    im = inp.squeeze(0).detach().cpu().numpy()
    im = np.moveaxis(im,0,-1)
    plt.imshow(im)

def srgb_to_rgb(srgb):
    ret = np.zeros_like(srgb)
    idx0 = srgb <= 0.04045
    idx1 = srgb > 0.04045
    ret[idx0] = srgb[idx0] / 12.92
    ret[idx1] = np.power((srgb[idx1] + 0.055) / 1.055, 2.4)
    return ret

# def rgb_to_chromaticity(rgb):
#     """ converts rgb to chromaticity """
#     irg = np.zeros_like(rgb)
#     s = np.sum(rgb, axis=-1) + 1e-6
#     irg[..., 0] = rgb[..., 0] / s
#     irg[..., 1] = rgb[..., 1] / s
#     irg[..., 2] = rgb[..., 2] / s
#     return irg



#!/usr/bin/env python
# coding: utf-8
res_metrics = {}
from argparse import ArgumentParser
import torch
import torchvision
import os
import torch.nn.functional as F
import torch.nn as nn
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
# from CGIntrinsics.models import networks
# from USI3D.usi_utils import get_config
# from USI3D.trainer import UnsupIntrinsicTrainer
import cv2
import numpy as np
import torchvision.utils as vutils
import numpy as np
import PIL
PIL.LOAD_TRUNCATED_IMAGES = True
import matplotlib.pyplot as plt
import glob
import sys
parser = ArgumentParser()
parser.add_argument("--model_type", default='cg')
parser.set_defaults(verbose=False)
from os import listdir
from os.path import isfile, join
from skimage.metrics import structural_similarity as ssim


def display_img_arr(img_arr, r, c, dim,titles_arr,display=True):
    fl = 0
    fig = plt.figure(figsize = dim)
    for i in range(r):
        for j in range(c):
            if len(img_arr) == fl:
                break
            ax1 = fig.add_subplot(r, c, fl + 1)
            ax1.axis("off")
            ax1.set_title(titles_arr[fl], fontsize = 20)
            ax1.imshow(img_arr[fl], cmap = 'gray')
            plt.subplots_adjust(left=0.125,
                bottom=0.1, 
                right=0.9, 
                top=0.9, 
                wspace=0.1, 
                hspace=0.1)
                
            fl = fl + 1
    if display:
        plt.show()
        
def srgb_to_rgb(srgb):
    ret = np.zeros_like(srgb)
    idx0 = srgb <= 0.04045
    idx1 = srgb > 0.04045
    ret[idx0] = srgb[idx0] / 12.92
    ret[idx1] = np.power((srgb[idx1] + 0.055) / 1.055, 2.4)
    return ret


def load_gray_image_from_file(filename, shape):
    try:
        # ensure image has no transparency channel
        img = cv2.cvtColor(cv2.imread(filename),cv2.COLOR_BGR2GRAY)   
        img = cv2.resize(img, (256,256))
        # Normalize pixel values to between 0 and 1.
        img = np.float32(img) / 255.0
#         print("max",img.max(),img.min())
        return img

    except Exception as e:
        print("eror in reading img", filename)
        return None
    return img


def load_mask_from_file(filename, shape,convert_srgb=False):
    try:
        # ensure image has no transparency channel
        img = cv2.imread(filename,0)
        img = cv2.resize(img, (256,256)) 
#         if img.max() > 100:
#             img = img/255.0
        # Normalize pixel values to between 0 and 1.
        return img
    except Exception as e:
        print("eror in reading img", filename)
        return None

# def load_image_from_file_(filename, shape,norm='nonorm',is_srgb=False):
#     # ensure image has no transparency channel
#     img = np.array(PIL.Image.open(open(filename, 'rb')).convert(
#         'RGB').resize(shape, PIL.Image.BILINEAR))
#     if img.max()==1.0:
#         norm = 'nonorm'
#     else:
#         norm = 'by255'
#     # Normalize pixel values to between 0 and 1.
#     if norm=='by255':
#         img = np.float32(img)/255.0
#     elif 'std_img':
#         img = (np.float32(img)-np.mean(img))/np.std(img)
#     elif 'std_imagenet':
#         img = (np.float32(img)-0.485)/0.229
#     elif 'nonorm':
#         pass
#     if is_srgb:
#         return srgb_to_rgb(img)
#     else:
#         return img
#     return img


def display_img_arr(img_arr, r, c, dim,titles_arr,display=True):
    fl = 0
    fig = plt.figure(figsize = dim)
    for i in range(r):
        for j in range(c):
            if len(img_arr) == fl:
                break
            print(img_arr[fl].max(),img_arr[fl].min(),titles_arr[fl])
            
            if display:
                ax1 = fig.add_subplot(r, c, fl + 1)
                ax1.axis("off")
                ax1.set_title(titles_arr[fl], fontsize = 20)
                ax1.imshow(img_arr[fl], cmap = 'gray')
                plt.subplots_adjust(left=0.125,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.1, 
                    hspace=0.1)
                
            fl = fl + 1
    if display:
        plt.show()


def srgb_to_rgb(srgb):
    ret = np.zeros_like(srgb)
    idx0 = srgb <= 0.04045
    idx1 = srgb > 0.04045
    ret[idx0] = srgb[idx0] / 12.92
    ret[idx1] = np.power((srgb[idx1] + 0.055) / 1.055, 2.4)
    return ret

def load_image_from_file_(filename, shape,norm='nonorm',is_srgb=False):
    # ensure image has no transparency channel
    img = np.array(PIL.Image.open(open(filename, 'rb')).convert(
        'RGB').resize(shape, PIL.Image.BILINEAR))
    if img.max()==1.0:
        norm = 'nonorm'
    else:
        norm = 'by255'
    # Normalize pixel values to between 0 and 1.
    if norm=='by255':
        img = np.float32(img)/255.0
    elif 'std_img':
        img = (np.float32(img)-np.mean(img))/np.std(img)
    elif 'std_imagenet':
        img = (np.float32(img)-0.485)/0.229
    elif 'nonorm':
        pass
    if is_srgb:
        return srgb_to_rgb(img)[:,:,:3]
    else:
        return img[:,:,:3]
    return img
    
def rgb_to_chromaticity(rgb):
    """ converts rgb to chromaticity """
    irg = np.zeros_like(rgb)
    s = np.sum(rgb, axis=-1) + 1e-6

    irg[..., 0] = rgb[..., 0] / s
    irg[..., 1] = rgb[..., 1] / s
    irg[..., 2] = rgb[..., 2] / s

    return irg


def get_np_imgs(R,S,color=None,path='./',name='',exp=True,normalise=False):
    if not(os.path.exists(path)):
        os.makedirs(path)
    R = R.squeeze(0)
    R = R.permute(1,2,0)
    r_np = R.detach().cpu().numpy()
    
    if normalise:
        r_np = (r_np - r_np.min())/(r_np.max()-r_np.min())
    if exp==True:
        r_np = np.exp(r_np)
    
    if S.shape[1]==3: #S predicted is colored
        S = S.squeeze(0)
        S = S.permute(1,2,0)
        s_np = S.detach().cpu().numpy()
        if exp==True:
            s_np = np.exp(s_np)
        if normalise:
            s_np = percentile_clipping(s_np,1,99)
            s_np = (s_np - s_np.min())/(s_np.max()-s_np.min())
        return r_np,s_np
    else:
        S = S.squeeze(0)
        s_np = S.squeeze(0).detach().cpu().numpy()
        if exp==True:
            s_np = np.exp(s_np)
        plt.imsave(os.path.join(path,'s'+name+'.png'),s_np,cmap='gray')

        if color is not None:
            col = color[0].detach().cpu().numpy()
            s_np_ = np.expand_dims(s_np,axis=2)
            new = np.concatenate((s_np_,s_np_,s_np_),axis=2)
            new[:,:,0] += col[0]
            new[:,:,1] += col[1]
            new[:,:,2] += col[2]
            new = np.exp(new).astype(float)
            new = (new- new.min())/(new.max() - new.min())
#             plt.imsave(os.path.join(path,'s_colored'+name+'.png'),new)
            return r_np,s_np,new

def min_max_norm(im):
    im = (im- im.min())/(im.max()-im.min())
    return im

def percentile_clipping(img1,l_per,u_per):
#     eq_img = np.zeros((img1.shape[0],img1.shape[1]),dtype = 'uint8')
    minn = np.percentile(img1,l_per)
    maxx = np.percentile(img1,u_per)
    img1[img1>maxx] = maxx
    img1[img1<minn] = minn
    return img1

def rgb_to_srgb(rgb):
    ret = torch.zeros_like(rgb)
    idx0 = rgb <= 0.0031308
    idx1 = rgb > 0.0031308
    ret[idx0] = rgb[idx0] * 12.92
    ret[idx1] = torch.pow(1.055 * rgb[idx1], 1.0 / 2.4) - 0.055
    return ret

def display_decomposed_(model,model_type, root, epoch, save_path='/media/Data2/avani.gupta/preds/ARAP_new_scenes' ,save_imgs=True):
    save_path = save_path +'/'+str(epoch)
    if not(os.path.exists(save_path)):
        os.makedirs(save_path)
    print(save_path)
    arap_path = os.path.join(root,'IID_data/ARAP_new_scenes/input/')
    imgs = glob.glob(arap_path+"*.png")+glob.glob(arap_path+"*.jpg")
    im,rgb_img,r,s,s_colored = None, None, None, None, None
    display = False

    for i in range(0,len(imgs)):
        name=imgs[i].split('/')[-1]
        
        filename = imgs[i]
        file = os.path.join(arap_path,filename)
        try:
            im = load_image_from_file_(file, shape=(256,256))
        except Exception as e:
            print("erorr {} in reading img {}".format(e,file))

        rgb_img = srgb_to_rgb(im)
        rgb_img[rgb_img < 1e-4] = 1e-4
        chromaticity = rgb_to_chromaticity(rgb_img)
        chromaticity = torch.from_numpy(np.transpose(chromaticity, (2,0,1))).contiguous().float().cuda()
        ex = torch.from_numpy(np.ascontiguousarray(np.transpose(im, (2,0,1)))).contiguous().float()
        ex.requires_grad = True

        if model_type=='cg':
            out = model(ex.unsqueeze(0).cuda())
            prediction_R, prediction_S = out
            R = torch.mul(chromaticity.unsqueeze(0),torch.exp(prediction_R))
            S = torch.exp(prediction_S)
            R = rgb_to_srgb(R.detach())
            if save_imgs:
                vsave_image(R.data, save_path+'/R'+str(name), padding=0, normalize=True)
                vsave_image(S.data, save_path+'/S'+str(name), padding=0, normalize=True)

        elif model_type=='wild':
            out = model(ex.unsqueeze(0).cuda())
            prediction_S, prediction_R, color = out
            R = torch.exp(prediction_R)
            R = rgb_to_srgb(R.detach())
            S = torch.exp(prediction_S)
            if save_imgs:
                vsave_image(R.data, save_path+'/R'+str(name), padding=0, normalize=True)
                vsave_image(S.data, save_path+'/S'+str(name), padding=0, normalize=True)

            S_colored = S
            S_colored = torch.cat((S_colored,S_colored,S_colored),axis=1)
            S_colored[:,0,:,:] += color[:,0]
            S_colored[:,1,:,:] += color[:,1]
            S_colored[:,2,:,:] += color[:,2]
            S_colored = rgb_to_srgb(S_colored.detach())
            if save_imgs:
                vsave_image(S_colored.data, save_path+'S_colored'+str(name), padding=0, normalize=True)

        elif model_type=='usi3d':
            with_mapping = True
            c_i, s_i_fake = gen_i.module.encode(ex.unsqueeze(0).cuda())
            if with_mapping:
                s_r, s_s = fea_s(s_i_fake)
            x_ri = gen_r.module.decode(c_i, s_r)
            x_si = gen_s.module.decode(c_i, s_s)
            R = (x_ri + 1) / 2.
            S = (x_si + 1) / 2.
            if save_imgs:
                vsave_image(R.data, save_path+'/R'+str(name), padding=0, normalize=True)
                vsave_image(S.data, save_path+'/S'+str(name), padding=0, normalize=True)
    print("saving images to",save_path)     
    return save_path

def get_mse_arap(model,model_type, root, epoch):
    pred_path = display_decomposed_(model,model_type,root,epoch)
    print("saved to ",pred_path)


    root = '/media/Data2/avani.gupta/IID_data'
    shape = (256,256)
    convert_srgb=False

    ssim_lis_r = []
    mse_lis_r = []
    lmse_lis_r = []
    psnr_lis_r = []
    ssim_bw_lis_r = []


    first = 1
    ssim_lis_s = []
    mse_lis_s = []
    lmse_lis_s = []
    psnr_lis_s = []
    ssim_bw_lis_s = []

    convert_srgb = False
    arap_path = root+'/ARAP_new_scenes/'
    ipath = arap_path+'input/'
    rpath = arap_path+'reflectance/'
    spath = arap_path+'shading/'
    mpath = arap_path+'mask/'
    cal_bwpsnr = Bandwise(partial(psnr_measure, data_range=255))
    cal_bwssim = Bandwise(partial(ssim, data_range=255))  

    ifiles = sorted([f for f in listdir(ipath) if isfile(join(ipath, f))])
    rfiles = sorted([f for f in listdir(rpath) if isfile(join(rpath, f))])
    sfiles = sorted([f for f in listdir(spath) if isfile(join(spath, f))])
    mfiles = sorted([f for f in listdir(mpath) if isfile(join(mpath, f))])

    for i,f in enumerate(ifiles):
        scene_light_setting = f.replace('.jpg','').replace('.png','')
        mfile_name = None
        light = None
        if 'light' in f:
            scene_name = f.split('_')[0]
        else:
            scene_name = f.replace('.jpg','').replace('.png','')

        if scene_name+'_alpha'+'.png' in mfiles:
            mfile_name = scene_name+'_alpha'+'.png'
        elif scene_name+'_alpha'+'.jpg' in mfiles:
            mfile_name = scene_name+'_alpha'+'.jpg'

        if mfile_name:   
            mask =  load_mask_from_file(os.path.join(mpath,mfile_name), shape=shape)
        else:
            print("creating mask of ones for scene", scene_name)
            mask = np.ones((shape[0],shape[1]))

        rfile_name = None
        if scene_name+'_albedo'+'.png' in rfiles:
            rfile_name = scene_name+'_albedo'+'.png'
        elif scene_name+'_albedo'+'.jpg' in rfiles:
            rfile_name = scene_name+'_albedo'+'.jpg'

        r_gt = load_image_from_file_(os.path.join(rpath,rfile_name), shape=shape)
        r_gt[:,:,0][mask==0.0] = 0.0
        r_gt[:,:,1][mask==0.0] = 0.0
        r_gt[:,:,2][mask==0.0] = 0.0

        r = load_image_from_file_(pred_path+'/R'+scene_light_setting+'.png', shape=shape)
        r[:,:,0][mask==0.0] = 0
        r[:,:,1][mask==0.0] = 0
        r[:,:,2][mask==0.0] = 0
        


        sfile_name = None
        if scene_light_setting+'_shading'+'.png' in sfiles:
            sfile_name = scene_light_setting+'_shading'+'.png'
        elif scene_name+'_shading'+'.jpg' in sfiles:
            sfile_name = scene_light_setting+'_shading'+'.jpg'

#         if model_type!='wild':
        s_gt = load_gray_image_from_file(os.path.join(spath,sfile_name), shape=shape)
        s_gt[mask==0.0] = 0.0
       

        s = load_gray_image_from_file(pred_path+'/S'+scene_light_setting+'.png', shape=shape)
        s[mask==0.0] = 0.0
        if len(r.shape)>3:
            r = r[:,:,:3]
            r_gt = r_gt[:,:,:3]

        lmse_r = local_error(r_gt, r,20,10)
        lmse_lis_r.append(lmse_r)
        psnr = np.mean(cal_bwpsnr(r_gt, r))
        ssim_bw = np.mean(cal_bwssim(r_gt, r))
        psnr_lis_r.append(psnr)
        # ssim_bw_lis_r.append(ssim_bw)

        mse = ((r - r_gt)**2).mean()
        mse_lis_r.append(mse)
        # breakpoint()
      
        ssim_lis_r.append(ssim(r,r_gt,data_range=r.max()-r.min(),channel_axis=2))

        mse = ((s - s_gt)**2).mean()
        mse_lis_s.append(mse)
        if len(s.shape)==2:
          s_ = np.expand_dims(s,axis=2)
          s_gt_ = np.expand_dims(s_gt,axis=2)
          lmse_s = local_error(s_gt_, s_, 20, 10)
          psnr = np.mean(cal_bwpsnr(s_gt_, s_))
          ssim_bw = np.mean(cal_bwssim(s_gt_, s_))
          
        else:
          lmse_s = local_error(s_gt, s, 20,10)
          psnr = np.mean(cal_bwpsnr(s_gt, ))
          ssim_bw = np.mean(cal_bwssim(s_gt, s))
        
        lmse_lis_s.append(lmse_s)
        psnr_lis_s.append(psnr)
        # ssim_bw_lis_s.append(ssim_bw)
        ssim_lis_s.append(ssim(s,s_gt,data_range=s.max()-s.min())) 

    # res_metrics= {}
    # res_metrics['ssim_R'], res_metrics['ssim_S'] = np.array(ssim_lis_r).mean(), np.array(ssim_lis_s).mean() 
    # res_metrics['mse_R'], res_metrics['mse_S'] = np.array(mse_lis_r).mean(), np.array(mse_lis_s).mean()    

    return np.array(mse_lis_r).mean(), np.array(mse_lis_s).mean(),np.array(ssim_lis_r).mean(), np.array(ssim_lis_s).mean(), np.array(lmse_lis_r).mean(), np.array(lmse_lis_s).mean(), np.array(psnr_lis_r).mean(), np.array(psnr_lis_s).mean()#, np.array(ssim_bw_lis_r).mean(),np.array(ssim_bw_lis_s).mean()

# from CGIntrinsics.data.data_loader import CreateDataLoaderIIWTest

def test_iiw(model, list_name, full_root,use_bilateral=False):
    total_loss = 0.0
    total_loss_eq = 0.0
    total_loss_ineq = 0.0
    total_count = 0.0
    # print("============================= Validation ============================")
    model.switch_to_eval()

    # for 3 different orientation
    for j in range(0,3):
        # print("============================= Testing EVAL MODE ============================", j)
        test_list_dir = full_root + '/CGIntrinsics/IIW/' + list_name
        print(test_list_dir)
        data_loader_IIW_TEST = CreateDataLoaderIIWTest(full_root, test_list_dir, j)
        dataset_iiw_test = data_loader_IIW_TEST.load_data()

        for i, data in enumerate(dataset_iiw_test):
            targets = data['target_1']
            stacked_img = data['img_1']
            # breakpoint()
            if use_bilateral:
              im_lis = []

              for k in range(len(stacked_img)):
                im = np.moveaxis(stacked_img[k].cpu().numpy(),0,-1)

                im = cv2.bilateralFilter(im, 15, 75, 75)
                im = cv2.bilateralFilter(im, 15, 75, 75)
                im = cv2.bilateralFilter(im, 15, 75, 75)
                im_lis.append(torch.from_numpy(np.moveaxis(im,-1,0)))
              
              stacked_img = torch.stack(im_lis)
            # print("stacked_img",stacked_img.shape)

            total_whdr, total_whdr_eq, total_whdr_ineq, count = model.evlaute_iiw(stacked_img, targets)
            total_loss += total_whdr
            total_loss_eq += total_whdr_eq
            total_loss_ineq += total_whdr_ineq

            total_count += count
            # print("Testing WHDR error ",j, i , total_loss/total_count)

    return total_loss/(total_count), total_loss_eq/total_count, total_loss_ineq/total_count

def display_decomposed_mit(model,model_type, root, epoch, sav_path='/media/Data2/avani.gupta/preds/MIT/' ,save_imgs=True,shape=(384,512)):
    sav_path = sav_path +str(epoch)
    if not(os.path.exists(sav_path)):
        os.makedirs(sav_path)
    print(sav_path)
    
    root = '/media/Data2/avani.gupta/IID_data'
    path = root+'/MIT-intrinsic/data/test/'
    img_folders = sorted(os.listdir(path))
    print(img_folders)
    for fol in img_folders:
        mask = load_mask_from_file(os.path.join(os.path.join(path,fol),'mask.png'), shape=shape)
        r_gt = load_gray_image_from_file(os.path.join(os.path.join(path,fol),'reflectance.png'), shape=shape)
    #     r_gt = cv2.cvtColor(r_gt, cv2.COLOR_BGR2GRAY)

        save_path = os.path.join(sav_path,fol)+'/'
        if not(os.path.exists(save_path)):
            os.makedirs(save_path)
        print(save_path)
    
        for i in range(0,11):
            if i==0:
                im = load_image_from_file(os.path.join(os.path.join(path,fol,'original.png')), shape=shape)  
                
            else:
                im = load_image_from_file(os.path.join(os.path.join(path,fol,'light'+str(i).zfill(2)+'.png')), shape=shape)  
            rgb_img = srgb_to_rgb(im)
            rgb_img[rgb_img < 1e-4] = 1e-4
            chromaticity = rgb_to_chromaticity(rgb_img)
            chromaticity = torch.from_numpy(np.transpose(chromaticity, (2,0,1))).contiguous().float().cuda()
            ex = torch.from_numpy(np.ascontiguousarray(np.transpose(im, (2,0,1)))).contiguous().float()
            ex.requires_grad = True
            name = fol+str(i)+'.png'
            if model_type=='cg':
                out = model(ex.unsqueeze(0).cuda())
                prediction_R, prediction_S = out
                R = torch.mul(chromaticity.unsqueeze(0),torch.exp(prediction_R))
                
                S = torch.exp(prediction_S)
                R = rgb_to_srgb(R.detach())
                if save_imgs:
                    print(i,"saving to",save_path+'R'+str(name))
                    vsave_image(R.data, save_path+'R'+str(name), padding=0, normalize=True)
                    vsave_image(S.data, save_path+'S'+str(name), padding=0, normalize=True)

            elif model_type=='wild':
                out = model(ex.unsqueeze(0).cuda())
                prediction_S, prediction_R, color = out
                R = torch.exp(prediction_R)
                R = rgb_to_srgb(R.detach())
                S = torch.exp(prediction_S)
                if save_imgs:
                    vsave_image(R.data, save_path+'R'+str(name), padding=0, normalize=True)
                    vsave_image(S.data, save_path+'S'+str(name), padding=0, normalize=True)

                S_colored = S
                S_colored = torch.cat((S_colored,S_colored,S_colored),axis=1)
                S_colored[:,0,:,:] += color[:,0]
                S_colored[:,1,:,:] += color[:,1]
                S_colored[:,2,:,:] += color[:,2]
                S_colored = rgb_to_srgb(S_colored.detach())
                if save_imgs:
                    vsave_image(S_colored.data, save_path+'S_colored'+str(name), padding=0, normalize=True)

            elif model_type=='usi3d':
                with_mapping = True
                c_i, s_i_fake = gen_i.module.encode(ex.unsqueeze(0).cuda())
                if with_mapping:
                    s_r, s_s = fea_s(s_i_fake)
                x_ri = gen_r.module.decode(c_i, s_r)
                x_si = gen_s.module.decode(c_i, s_s)
                R = (x_ri + 1) / 2.
                S = (x_si + 1) / 2.
                if save_imgs:
                    vsave_image(R.data, save_path+'R'+str(name), padding=0, normalize=True)
                    vsave_image(S.data, save_path+'S'+str(name), padding=0, normalize=True)

    return sav_path


def get_mse_mit(model,model_type, root, epoch):
    path = root+'/IID_data/MIT-intrinsic/data/test/'
    img_folders = os.listdir(path)
    pred_path = display_decomposed_mit(model,model_type,root,epoch)

    ssim_dic_r = {}
    mse_dic_r = {}
    ssim_dic_s = {}
    mse_dic_s = {}

    shape = (256,256)
    convert_srgb=False

    ssim_lis_r = []
    mse_lis_r = []
    lmse_lis_r = []

    first = 1
    ssim_lis_s = []
    mse_lis_s = []
    lmse_lis_s = []

    for fol in img_folders:
        for i in range(11):
            r_gt = load_image_from_file(os.path.join(os.path.join(path,fol),'reflectance.png'), shape=shape)
            mask = load_mask_from_file(os.path.join(os.path.join(path,fol),'mask.png'), shape=shape)
            
            if i==0:
                s_gt = load_gray_image_from_file((os.path.join(path,fol)+'/shading.png'), shape=shape)
                s = load_gray_image_from_file(pred_path+fol+'/S'+fol+str(i)+'.png', shape=shape)
                r = load_image_from_file(pred_path+fol+'/R'+fol+str(i)+'.png', shape=shape)
            else:
                s_gt = load_gray_image_from_file(os.path.join(os.path.join(path,fol),'calc_shadinglight'+str(i).zfill(2)+'.png'), shape=shape)
                s = load_gray_image_from_file(pred_path+fol+'/S'+fol+str(i)+'.png', shape=shape)
                r = load_image_from_file(pred_path+fol+'/R'+fol+str(i)+'.png', shape=shape)
            # breakpoint()
            # print("bef",r.max(),r.min(),r_gt.max(),r_gt.min(),s.max(),s.min(),s_gt.max(),s_gt.min())
            r_gt[:,:,0][mask==0] = 0
            r_gt[:,:,1][mask==0] = 0
            r_gt[:,:,2][mask==0] = 0  
            
            r[:,:,0][mask==0] = 0
            r[:,:,1][mask==0] = 0
            r[:,:,2][mask==0] = 0  
            s[mask==0] = 0
            s_gt[mask==0] = 0
            
            
            print("after",r.max(),r.min(),r_gt.max(),r_gt.min(),s.max(),s.min(),s_gt.max(),s_gt.min())
            if i==0:
                plt.imsave(pred_path+'S'+fol+'original.png',s)
                plt.imsave(pred_path+'R'+fol+'original.png',r)
            else:
                plt.imsave(pred_path+'S'+fol+'light'+str(i).zfill(2)+'.png',s)
                plt.imsave(pred_path+'R'+fol+'light'+str(i).zfill(2)+'.png',r)
            
#             display_img_arr([s,s_gt,r,r_gt],1,4,(16,16), [model_type+'s','s_gt','r','r_gt'])

        
        mse = ((r - r_gt)**2).mean()
        mse_lis_r.append(mse)
        ssim_lis_r.append(ssim(r,r_gt,data_range=r.max()-r.min(),channel_axis=2))

        mse = ((s - s_gt)**2).mean()
        mse_lis_s.append(mse)

        
        ssim_lis_s.append(ssim(s,s_gt,data_range=s.max()-s.min())) 

    # res_metrics= {}
    # res_metrics['ssim_R'], res_metrics['ssim_S'] = np.array(ssim_lis_r).mean(), np.array(ssim_lis_s).mean() 
    # res_metrics['mse_R'], res_metrics['mse_S'] = np.array(mse_lis_r).mean(), np.array(mse_lis_s).mean()    

    return np.array(mse_lis_r).mean(), np.array(mse_lis_s).mean(),np.array(ssim_lis_r).mean(), np.array(ssim_lis_s).mean()