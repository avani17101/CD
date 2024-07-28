from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torchvision import transforms
from .data import ImageFilelist, ImageFolder, CGIFolder, MPIFolder
import torch
import os
import math
import torchvision.utils as vutils
import yaml
import numpy as np
import torch.nn.init as init
import time
import tensorflow as tf


def get_local_time():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def get_intrinsic_data_loader(conf, is_sup=False, rate=None):
    if rate is None:
        rate = ''
    else:
        rate = '-{}'.format(rate)
    if is_sup:
        train_dict = {'input': 'train-input_sup.txt',
                      'albedo': 'train-reflectance_sup.txt',
                      'shading': 'train-shading_sup.txt',
                      'mask': 'train-mask_sup.txt'}
    else:
        train_dict = {'input': 'train-input.txt',
                      'albedo': 'train-reflectance{}.txt'.format(rate),
                      'shading': 'train-shading{}.txt'.format(rate),
                      'mask': 'train-mask.txt'}
    test_dict = {'input': 'test-input.txt',
                 'albedo': 'test-reflectance.txt',
                 'shading': 'test-shading.txt',
                 'mask': 'test-mask.txt'}

    if conf['dataset_name'] == 'CGI':
        train_set = CGIFolder(conf, dict_image=train_dict, is_train=True)
        test_set = CGIFolder(conf, dict_image=test_dict, is_train=False)
    elif 'MIT' in conf['dataset_name']:
        train_set = CGIFolder(conf, dict_image=train_dict, is_train=True)
        test_set = CGIFolder(conf, dict_image=test_dict, is_train=False)
    elif 'ShapeNet' in conf['dataset_name']:
        train_set = CGIFolder(conf, dict_image=train_dict, is_train=True)
        test_set = CGIFolder(conf, dict_image=test_dict, is_train=False)
    elif 'MIX' in conf['dataset_name']:
        train_set = CGIFolder(conf, dict_image=train_dict, is_train=True)
        test_set = CGIFolder(conf, dict_image=test_dict, is_train=False)
    else:
        raise ValueError('dataset name error in config file: {}'.format(conf['dataset_name']))

    train_loader = DataLoader(train_set, num_workers=conf['num_workers'], shuffle=True)
    test_loader = DataLoader(test_set, num_workers=conf['num_workers'], shuffle=False)

    return train_loader, test_loader


def get_all_data_loaders(conf):
    batch_size = conf['batch_size']
    num_workers = conf['num_workers']
    if 'new_size' in conf:
        new_size_a = new_size_b = conf['new_size']
    else:
        new_size_a = conf['new_size_a']
        new_size_b = conf['new_size_b']
    height = conf['crop_image_height']
    width = conf['crop_image_width']

    train_loader_a = get_data_loader_folder(os.path.join(conf['data_root'], 'trainA'), batch_size, True,
                                            new_size_a, height, width, num_workers, True)
    test_loader_a = get_data_loader_folder(os.path.join(conf['data_root'], 'testA'), 1, False,
                                           new_size_a, new_size_a, new_size_a, num_workers, False)
    train_loader_b = get_data_loader_folder(os.path.join(conf['data_root'], 'trainB'), batch_size, True,
                                            new_size_b, height, width, num_workers, True)
    test_loader_b = get_data_loader_folder(os.path.join(conf['data_root'], 'testB'), 1, False,
                                           new_size_b, new_size_b, new_size_b, num_workers, False)
    train_loader_c = get_data_loader_folder(os.path.join(conf['data_root'], 'trainC'), batch_size, True,
                                            new_size_b, height, width, num_workers, True)
    test_loader_c = get_data_loader_folder(os.path.join(conf['data_root'], 'testC'), 1, False,
                                           new_size_b, new_size_b, new_size_b, num_workers, False)

    return train_loader_a, train_loader_b, train_loader_c, test_loader_a, test_loader_b, test_loader_c


def get_data_loader_list(root, file_list, batch_size, train, new_size=None,
                         height=256, width=256, num_workers=4, crop=True):
    transform_list = [transforms.ToTensor(),
                      transforms.Normalize((0.5, 0.5, 0.5),
                                           (0.5, 0.5, 0.5))]
    transform_list = [transforms.RandomCrop((height, width))] + transform_list if crop else transform_list
    transform_list = [transforms.Resize(new_size)] + transform_list if new_size is not None else transform_list
    transform_list = [transforms.RandomHorizontalFlip()] + transform_list if train else transform_list
    transform = transforms.Compose(transform_list)
    dataset = ImageFilelist(root, file_list, transform=transform)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=train, drop_last=True, num_workers=num_workers)
    return loader


def get_data_loader_folder(input_folder, batch_size, train, new_size=None,
                           height=256, width=256, num_workers=4, crop=True):
    transform_list = [transforms.ToTensor(),
                      transforms.Normalize((0.5, 0.5, 0.5),
                                           (0.5, 0.5, 0.5))]
    transform_list = [transforms.RandomCrop((height, width))] + transform_list if crop else transform_list
    transform_list = [transforms.Resize(new_size)] + transform_list if new_size is not None else transform_list
    transform_list = [transforms.RandomHorizontalFlip()] + transform_list if train else transform_list
    transform = transforms.Compose(transform_list)
    dataset = ImageFolder(input_folder, transform=transform)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=train, drop_last=True, num_workers=num_workers)
    return loader


def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream)


def eformat(f, prec):
    s = "%.*e" % (prec, f)
    mantissa, exp = s.split('e')
    # add 1 to digits as 1 is taken by sign +/-
    return "%se%d" % (mantissa, int(exp))


def __write_images(image_outputs, display_image_num, file_name):
    image_outputs = [images.expand(-1, 3, -1, -1) for images in image_outputs]  # expand gray-scale images to 3 channels
    image_tensor = torch.cat([images[:display_image_num] for images in image_outputs], 0)
    image_grid = vutils.make_grid(image_tensor.data, nrow=display_image_num, padding=0, normalize=True)
    vutils.save_image(image_grid, file_name, nrow=1)


def write_2images(image_outputs, display_image_num, image_directory, postfix):
    n = len(image_outputs)
    if n == 10:
        __write_images(image_outputs[:2], display_image_num, '%s/gen_I(I-I_recon)_%s.jpg' % (image_directory, postfix))
        __write_images(image_outputs[2: 6], display_image_num,
                       '%s/gen_R(R-R_recon-R_s-R_i)_%s.jpg' % (image_directory, postfix))
        __write_images(image_outputs[6:], display_image_num,
                       '%s/gen_S(S-S_recon-S_r-S_i)_%s.jpg' % (image_directory, postfix))
    elif n == 5:
        __write_images(image_outputs, display_image_num, '%s/I-R_gt-R_pred-S_gt-S_pred-%s.jpg' % (image_directory, postfix))
    else:
        raise ValueError('n should be 5, or 10, but is {}'.format(n))


def prepare_sub_folder(output_directory):
    image_directory = os.path.join(output_directory, 'images')
    if not os.path.exists(image_directory):
        print("Creating directory: {}".format(image_directory))
        os.makedirs(image_directory)
    checkpoint_directory = os.path.join(output_directory, 'checkpoints')
    if not os.path.exists(checkpoint_directory):
        print("Creating directory: {}".format(checkpoint_directory))
        os.makedirs(checkpoint_directory)
    return checkpoint_directory, image_directory


def write_one_row_html(html_file, iterations, img_filename, all_size):
    html_file.write("<h3>iteration [%d] (%s)</h3>" % (iterations, img_filename.split('/')[-1]))
    html_file.write("""
        <p><a href="%s">
          <img src="%s" style="width:%dpx">
        </a><br>
        <p>
        """ % (img_filename, img_filename, all_size))
    return


def write_html(filename, iterations, image_save_iterations, image_directory, all_size=1536):
    html_file = open(filename, "w")
    html_file.write('''
    <!DOCTYPE html>
    <html>
    <head>
      <title>Experiment name = %s</title>
      <meta http-equiv="refresh" content="30">
    </head>
    <body>
    ''' % os.path.basename(filename))
    html_file.write("<h3>current</h3>")
    write_one_row_html(html_file, iterations, '%s/gen_a2b_train_current.jpg' % (image_directory), all_size)
    write_one_row_html(html_file, iterations, '%s/gen_b2a_train_current.jpg' % (image_directory), all_size)
    for j in range(iterations, image_save_iterations - 1, -1):
        if j % image_save_iterations == 0:
            write_one_row_html(html_file, j, '%s/gen_a2b_test_%08d.jpg' % (image_directory, j), all_size)
            write_one_row_html(html_file, j, '%s/gen_b2a_test_%08d.jpg' % (image_directory, j), all_size)
            write_one_row_html(html_file, j, '%s/gen_a2b_train_%08d.jpg' % (image_directory, j), all_size)
            write_one_row_html(html_file, j, '%s/gen_b2a_train_%08d.jpg' % (image_directory, j), all_size)
    html_file.write("</body></html>")
    html_file.close()


def write_loss(iterations, trainer, train_writer):
    members = [attr for attr in dir(trainer) \
               if not callable(getattr(trainer, attr)) and not attr.startswith("__") and (
                       'loss' in attr or 'grad' in attr or 'nwd' in attr)]
    for m in members:
        train_writer.add_scalar(m, getattr(trainer, m), iterations + 1)


def slerp(val, low, high):
    """
    original: Animating Rotation with Quaternion Curves, Ken Shoemake
    https://arxiv.org/abs/1609.04468
    Code: https://github.com/soumith/dcgan.torch/issues/14, Tom White
    """
    omega = np.arccos(np.dot(low / np.linalg.norm(low), high / np.linalg.norm(high)))
    so = np.sin(omega)
    return np.sin((1.0 - val) * omega) / so * low + np.sin(val * omega) / so * high


def get_slerp_interp(nb_latents, nb_interp, z_dim):
    """
    modified from: PyTorch inference for "Progressive Growing of GANs" with CelebA snapshot
    https://github.com/ptrblck/prog_gans_pytorch_inference
    """

    latent_interps = np.empty(shape=(0, z_dim), dtype=np.float32)
    for _ in range(nb_latents):
        low = np.random.randn(z_dim)
        high = np.random.randn(z_dim)  # low + np.random.randn(512) * 0.7
        interp_vals = np.linspace(0, 1, num=nb_interp)
        latent_interp = np.array([slerp(v, low, high) for v in interp_vals],
                                 dtype=np.float32)
        latent_interps = np.vstack((latent_interps, latent_interp))

    return latent_interps[:, :, np.newaxis, np.newaxis]


# Get model list for resume
def get_model_list(dirname, key):
    if os.path.exists(dirname) is False:
        return None
    gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
                  os.path.isfile(os.path.join(dirname, f)) and key in f and ".pt" in f]
    if len(gen_models) == 0:
        return None
    gen_models.sort()
    last_model_name = gen_models[-1]
    return last_model_name


def get_scheduler(optimizer, hyperparameters, iterations=-1):
    if 'lr_policy' not in hyperparameters or hyperparameters['lr_policy'] == 'constant':
        scheduler = None  # constant scheduler
    elif hyperparameters['lr_policy'] == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=hyperparameters['step_size'],
                                        gamma=hyperparameters['gamma'], last_epoch=iterations)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', hyperparameters['lr_policy'])
    return scheduler


def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            # print m.__class__.__name__
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    return init_fun


class Timer:
    def __init__(self, msg):
        self.msg = msg
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_value, exc_tb):
        print(self.msg % (time.time() - self.start_time))
        


def make_dir_if_not_exists(directory):
  if not tf.io.gfile.exists(directory):
    tf.io.gfile.makedirs(directory)
    
import numpy as np
from scipy.stats import ttest_ind


def flatten(nested_list):
    """Flatten a nested list."""
    return [item for a_list in nested_list for item in a_list]


def process_what_to_run_expand(pairs_to_test,
                               random_counterpart=None,
                               num_random_exp=100,
                               random_concepts=None):
    """Get concept vs. random or random vs. random pairs to run.

      Given set of target, list of concept pairs, expand them to include
       random pairs. For instance [(t1, [c1, c2])...] becomes
       [(t1, [c1, random1],
        (t1, [c1, random2],...
        (t1, [c2, random1],
        (t1, [c2, random2],...]

    Args:
      pairs_to_test: [(target, [concept1, concept2,...]),...]
      random_counterpart: random concept that will be compared to the concept.
      num_random_exp: number of random experiments to run against each concept.
      random_concepts: A list of names of random concepts for the random
                       experiments to draw from. Optional, if not provided, the
                       names will be random500_{i} for i in num_random_exp.

    Returns:
      all_concepts: unique set of targets/concepts
      new_pairs_to_test: expanded
    """

    def get_random_concept(i):
        return (random_concepts[i] if random_concepts
                else 'random500_{}'.format(i))

    new_pairs_to_test = []
    for (target, concept_set) in pairs_to_test:
        new_pairs_to_test_t = []
        # if only one element was given, this is to test with random.
        if len(concept_set) == 1:
            i = 0
            while len(new_pairs_to_test_t) < min(100, num_random_exp):
                # make sure that we are not comparing the same thing to each other.
                if concept_set[0] != get_random_concept(
                        i) and random_counterpart != get_random_concept(i):
                    new_pairs_to_test_t.append(
                        (target, [concept_set[0], get_random_concept(i)]))
                i += 1
        elif len(concept_set) > 1:
            new_pairs_to_test_t.append((target, concept_set))
        else:
            tf.logging.info('PAIR NOT PROCCESSED')
        new_pairs_to_test.extend(new_pairs_to_test_t)

    all_concepts = list(set(flatten([cs + [tc] for tc, cs in new_pairs_to_test])))

    return all_concepts, new_pairs_to_test


def process_what_to_run_concepts(pairs_to_test):
    """Process concepts and pairs to test.

    Args:
      pairs_to_test: a list of concepts to be tested and a target (e.g,
       [ ("target1",  ["concept1", "concept2", "concept3"]),...])

    Returns:
      return pairs to test:
         target1, concept1
         target1, concept2
         ...
         target2, concept1
         target2, concept2
         ...

    """

    pairs_for_sstesting = []
    # prepare pairs for concpet vs random.
    for pair in pairs_to_test:
        for concept in pair[1]:
            pairs_for_sstesting.append([pair[0], [concept]])
    return pairs_for_sstesting


def process_what_to_run_randoms(pairs_to_test, random_counterpart):
    """Process concepts and pairs to test.

    Args:
      pairs_to_test: a list of concepts to be tested and a target (e.g,
       [ ("target1",  ["concept1", "concept2", "concept3"]),...])
      random_counterpart: a random concept that will be compared to the concept.

    Returns:
      return pairs to test:
            target1, random_counterpart,
            target2, random_counterpart,
            ...
    """
    # prepare pairs for random vs random.
    pairs_for_sstesting_random = []
    targets = list(set([pair[0] for pair in pairs_to_test]))
    for target in targets:
        pairs_for_sstesting_random.append([target, [random_counterpart]])
    return pairs_for_sstesting_random


# helper functions to write summary files
def print_results(results, random_counterpart=None, random_concepts=None, num_random_exp=100,
                  min_p_val=0.05):
    """Helper function to organize results.
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
    print("Class =", results[0]['target_class'])

    # prepare data
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

    # print concepts and classes with indentation
    for concept in result_summary:

        # if not random
        if not is_random_concept(concept):
            print(" ", "Concept =", concept)

            for bottleneck in result_summary[concept]:
                i_ups = [item['i_up'] for item in result_summary[concept][bottleneck]]

                # Calculate statistical significance
                _, p_val = ttest_ind(random_i_ups[bottleneck], i_ups)

                print(3 * " ", "Bottleneck =", ("%s. TCAV Score = %.2f (+- %.2f), "
                                                "random was %.2f (+- %.2f). p-val = %.3f (%s)") % (
                          bottleneck, np.mean(i_ups), np.std(i_ups),
                          np.mean(random_i_ups[bottleneck]),
                          np.std(random_i_ups[bottleneck]), p_val,
                          "not significant" if p_val > min_p_val else "significant"))







