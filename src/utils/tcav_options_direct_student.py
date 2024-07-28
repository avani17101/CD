import argparse
import os

class TCAVOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--sample_type',default='0.5',type=str)
        self.parser.add_argument('--teacher_type',default='dino',type=str) #dino_vitsmall
        self.parser.add_argument('--map_type',default='T_to_S',type=str)
        self.parser.add_argument('--model_type',default='decoymnist',type=str)
        self.parser.add_argument('--dset',default='dmnist',type=str)
        self.parser.add_argument('--bottleneck_name',default='conv2',type=str)
        self.parser.add_argument('--freeze_bef_layers', type=int, default=0, help='whther to load best finetuned model')

        self.parser.add_argument('--cav_update_freq', type=float, default=150)
        self.parser.add_argument("--random_set", default=2, type=int)
        self.parser.add_argument("--use_one_cav_per_con", default=1, type=int)
        self.parser.add_argument("--dy_fc_learn", default=1, type=int)
        self.parser.add_argument("--run_num", default=0, type=int)
        self.parser.add_argument("--run_random", default=0, type=int)
        self.parser.add_argument("--write", default=True, type=bool)
        self.parser.add_argument("--save_path", default='/media/Data2/avani.gupta/')
        self.parser.add_argument("--concepts_finetune", default='both')#option=['albedo_finetune','ill_finetune'])
        self.parser.add_argument("--nrm", default=False)
        self.parser.add_argument("--trainable", default=True)
        self.parser.add_argument("--root", default='/media/Data2/avani.gupta/IID_data')
        self.parser.add_argument("--finetune", default=0, type=int, help="finetune using tcav loss or not")
        self.parser.add_argument("--set_to_ep_level", default=0, type=int, help="set cav updation freq to epoch level")
        self.parser.add_argument("--get_tcav_score",default=0, type=int, help="finetune using tcav loss or not")
        self.parser.add_argument("--last_layer", default=1, type=int, help="whether bottleneck is last layer or not")
        self.parser.add_argument("--test_code",default=0, type=int, help="just for code testing")
        self.parser.add_argument("--loss_type",default='L1_cos', type=str, help='which type of tcav loss to use')
        self.parser.add_argument("--use_nn_cav",default=False, type=int, help='lr or nn')
        self.parser.add_argument("--num_random_exp",default=1, type=int, help='number of random experiments')
        self.parser.add_argument("--epochs",default=50, type=int, help='number of epochs to train upon')
        self.parser.add_argument("--train_with_orig_loss",default=1, type=int, help='whether add conventional GT loss or not')
        self.parser.add_argument("--use_triplet_loss",default=1, type=int, help='use triplet loss to push input(anchor) to desired concept direction (pos) or (neg)')
        self.parser.add_argument("--use_cav_loss",default=1, type=int, help='use cav loss or not')
        self.parser.add_argument("--reg_type",default='svm', type=str, help='decision boundary by logistic or linear reg')
        self.parser.add_argument("--nn_dense",default='normal', type=str, help='increase layers in nn')
        self.parser.add_argument("--use_wandb",default=1, type=int)
        self.parser.add_argument("--wtcav",default=5, type=float)
        self.parser.add_argument("--swtcav",default=0.5, type=float)
        self.parser.add_argument("--use_clarc", default=0,help='whether to use clarc or not')
        self.parser.add_argument("--num_concepts", default=10,type=int, help='number of concepts to optimize over in one cav iter')
        self.parser.add_argument("--revise_cav_iters", default=1,type=int, help='number of iters to retrain cav upon')
        # self.parser.add_argument("--sample_concepts", s=0,type=int, help='whether to sample concepts for cav training or train over all in each epoch')
        self.parser.add_argument("--concept_train_start_ep", default=1,type=int, help='which epoch to start cav training from')
        self.parser.add_argument("--train_from_scratch", default=0, type=int,help="whether to train from scratch or not")
        self.parser.add_argument("--batch_size", default=64, type=int,help="batch size")
        self.parser.add_argument("--smoothness_finetune", default=0, type=int,help="whether to finetune for smoothness")
        self.parser.add_argument("--wtl", default=0.01, type=float,help="wt of tl")
        self.parser.add_argument("--use_gt_loss", default=1, type=int,help="whether to train without gt or not")
        self.parser.add_argument("--test", default=1, type=int,help="whether to test for metrics(whdr, mse, ssim or not)")
        self.parser.add_argument("--gpu_ids",default="0", type=str, help='gpu ids')
        self.parser.add_argument('--checkpoints_dir', type=str, default='/media/Data2/avani.gupta/new_checkpoints/', help='models are saved here')
        self.parser.add_argument('--name', type=str, default='test_local', help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--teach_mapped_to_stu', type=int, default=1, help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--do_proto_mean', type=int, default=1, help='whether to do the proto mean thingy')
        self.parser.add_argument('--cur_proto_mean_wt', type=float, default=0.3, help='what proto')
        self.parser.add_argument('--cav_update_wt', type=float, default=0.3, help='what wt of cav updates')
        self.parser.add_argument('--num_cons', type=int, default=1, help='num of cons')
        self.parser.add_argument('--barlow_lamb', type=float, default=0.5, help='num of cons')
        self.parser.add_argument('--wp', type=float, default=0.5, help='num of cons')
        self.parser.add_argument("--mapping_mod_epoch", default=5, type=int, help="whuch ep mapping module to load")
        self.parser.add_argument("--update_cavs", default=0, type=int, help="whether to update cavs or not")
        # self.parser.add_argument('--dnum', type=int, default=4, help='pair type 2,3,4')
        # self.parser.add_argument('--use_multi_pairs', type=int, default=0, help="whther to use multiple pairs")
        self.parser.add_argument('--pairs_vals', type=str, default="5", help="which multiple pairs")
        self.parser.add_argument('--pair_affect', type=str, default="0", help="which multiple pairs")
        self.parser.add_argument('--num_imgs', type=int, default=150, help='num of imgs')
        # self.parser.add_argument('--affect', type=int, default=0, help='affect 1 or not affect 0')
        self.parser.add_argument('--use_knn_proto', type=int, default=1, help='whether to use knn protos')
        self.parser.add_argument('--knn_k', type=int, default=7, help='k in knn')
        self.parser.add_argument('--class_wise_training', type=int, default=0, help='whther to do class specific training')
        self.parser.add_argument('--use_proto', type=int, default=1, help='whther to use prototypes as gt for training')
        self.parser.add_argument('--use_last_layer_proto', type=int, default=0, help='whther to use prototypes as labels in last layer as gt for training')
        self.parser.add_argument('--update_proto', type=int, default=1, help='whther to update protos or use  the same one as init')
        self.parser.add_argument('--use_precalc_proto', type=int, default=1, help='whther to use precalculated proto during start')
        self.parser.add_argument('--load_best_finetuned', type=int, default=0, help='whther to load best finetuned model')
        self.parser.add_argument('--without_distil', type=int, default=0, help='whther to do without distil')
        self.parser.add_argument('--cdep_grad_method', type=int, default=2, help='grad method cdep')
        self.parser.add_argument('--use_cdepcolor', type=int, default=1, help='whether to use cdep color')
        self.parser.add_argument('--regularizer_rate', type=float, default=0.3, help='regularizer rate for cdep')
        self.parser.add_argument("--bias", default=1, type=int, help="bias type 1 or 0")
        #clever_hans
        # self.parser.add_argument("--name",default=datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),help="Name to store the log file as",)
        self.parser.add_argument("--seed", type=int, default=42, help="Random generator seed for all frameworks")
        self.parser.add_argument("--resume", help="Path to log file to resume from")
        self.parser.add_argument("--mode", default="train", help="train, test, or plot")
        # self.parser.add_argument("--data-dir", default="/media/Data2/avani.gupta/CLEVR-Hans3/", help="Directory to data")
        # self.parser.add_argument("--fp-ckpt", type=str, default='/home/avani.gupta/tcav_pt/NeSyXIL/src/clevr_hans/cnn/runs/conf_3/resnet-clevr-hans-17-conf_3_seed0/model_epoch56_bestvalloss_0.0132.pth', help="checkpoint filepath")
        # self.parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train with")
        self.parser.add_argument("--lr", type=float, default=1e-3, help="Outer learning rate of model")
        # self.parser.add_argument("--batch-size", type=int, default=32, help="Batch size to train with")
        self.parser.add_argument("--num-workers", type=int, default=4, help="Number of threads for data loader")
        self.parser.add_argument("--dataset", default = "clevr-hans-state", choices=["clevr-hans-state"],)
        self.parser.add_argument("--no-cuda", action="store_true", help="Run on CPU instead of GPU (not recommended)",)
        self.parser.add_argument("--train-only", action="store_true", help="Only run training, no evaluation")
        self.parser.add_argument("--eval-only", action="store_true", help="Only run evaluation, no training")
        # self.parser.add_argument("--conf_version", default="CLEVR-Hans3", help="conf version")
       



    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >=0:
                self.opt.gpu_ids.append(id)

        args = vars(self.opt)

        #clever_hans
        # args.conf_version = args.data_dir.split(os.path.sep)[-2]
        # args.name = args.name + f"-{args.conf_version}"
        #clever_hans end

        print('------------ Options ------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        expr_dir =  os.path.join(self.opt.checkpoints_dir, self.opt.name)
        if not os.path.exists(expr_dir):
            os.makedirs(expr_dir)

        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt
