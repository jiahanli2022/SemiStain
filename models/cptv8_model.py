import numpy as np
import torch
from models.asp_loss import AdaptiveSupervisedPatchNCELoss
from .base_model import BaseModel
from . import networks
from .patchnce import PatchNCELoss
from .gauss_pyramid import Gauss_Pyramid_Conv
import util.util as util
from copy import deepcopy
import torch.nn.functional as F
import cv2

class MultiLSHRandomProjection:
    def __init__(self, d_model, b, device, num_tables=10, dis_threshold=0):
        self.b = b
        self.num_tables = num_tables
        self.dis_threshold = dis_threshold
        self.device = device
        self.Rs = [torch.randn(d_model, b // 2).to(device) for _ in range(num_tables)]
        
    
    def hash(self, x, R):
        proj = x @ R
        neg_proj = -proj
        
        concat_proj = torch.cat([proj, neg_proj], dim=-1)
        return torch.argmax(concat_proj, dim=-1)

    def cluster(self, X):
        batch_size = X.shape[0]
        hash_signatures = torch.zeros((batch_size, self.num_tables), dtype=torch.float).to(self.device)
        
        for i, R in enumerate(self.Rs):
            hash_signatures[:, i] = self.hash(X, R)

        cluster_labels = self._hamming_clustering(hash_signatures)

        return cluster_labels

    def _hamming_clustering(self, hash_signatures):
        num_samples = hash_signatures.shape[0]
        cluster_labels = torch.full((num_samples,), -1, dtype=torch.long).to(self.device)
        
        cluster_idx = 0

        for i in range(num_samples):
            if cluster_labels[i] != -1:
                continue

            distances = torch.sum(hash_signatures != hash_signatures[i], dim=-1)
            
            similar_samples = torch.where(distances <= self.dis_threshold)[0]
            
            cluster_labels[i] = cluster_idx

            for sample_idx in similar_samples:
                if cluster_labels[sample_idx] == -1:
                    cluster_labels[sample_idx] = cluster_idx

            cluster_idx += 1
        
        return cluster_labels.view(-1, 1)


class CPTV8Model(BaseModel):
    """ Contrastive Paired Translation (CPT).
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for CUT model
        """
        parser.add_argument('--CUT_mode', type=str, default="CUT", choices='(CUT, cut, FastCUT, fastcut)')

        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss: GAN(G(X))')
        parser.add_argument('--lambda_NCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--nce_idt', type=util.str2bool, nargs='?', const=True, default=False, help='use NCE loss for identity mapping: NCE(G(Y), Y))')
        parser.add_argument('--nce_layers', type=str, default='0,4,8,12,16', help='compute NCE loss on which layers')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
        parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'], help='how to downsample the feature map')
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--nce_LSH', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--flip_equivariance',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT")
        parser.set_defaults(pool_size=0)  # no image pooling

        # FDL:
        parser.add_argument('--lambda_gp', type=float, default=1.0, help='weight for Gaussian Pyramid reconstruction loss')
        parser.add_argument('--gp_weights', type=str, default='uniform', help='weights for reconstruction pyramids.')
        parser.add_argument('--num_tables', type=int, default=5, help='num for space')
        parser.add_argument('--LSH_scalers', type=str)
        parser.add_argument('--fixed_cluster', type=str, default='True')
        parser.add_argument('--lambda_asp', type=float, default=0.0, help='weight for ASP loss')
        parser.add_argument('--asp_loss_mode', type=str, default='none', help='"scheduler_lookup" options for the ASP loss. Options for both are listed in Fig. 3 of the paper.')
        parser.add_argument('--n_downsampling', type=int, default=2, help='# of downsample in G')

        parser.add_argument('--lambda_consist', type=float, default=0.0, help='weight for Consistency loss')

        opt, _ = parser.parse_known_args()

        assert opt.fixed_cluster in ['True', 'False']

        # Set default parameters for CUT and FastCUT
        if opt.CUT_mode.lower() == "cut":
            parser.set_defaults(nce_idt=True, lambda_NCE=1.0)
        elif opt.CUT_mode.lower() == "fastcut":
            parser.set_defaults(
                nce_idt=False, lambda_NCE=10.0, flip_equivariance=False,
                n_epochs=20, n_epochs_decay=10
            )
        else:
            raise ValueError(opt.CUT_mode)

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G', 'NCE']
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]

        if opt.nce_idt and self.isTrain:
            self.loss_names += ['NCE_Y']
            self.visual_names += ['idt_B']

        if self.isTrain:
            self.model_names = ['G', 'F', 'D']
            if self.opt.use_semistain:
                self.model_names.append('G_EMA')
        else:
            self.model_names = ['G_EMA']

        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)
        self.netF = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
        if self.opt.use_semistain:
            self.netG_EMA = deepcopy(self.netG)

        if self.isTrain:
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionNCE = PatchNCELoss(opt).to(self.device)

            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            if self.opt.lambda_gp > 0:
                self.P = Gauss_Pyramid_Conv(num_high=5)
                self.criterionGP = torch.nn.L1Loss().to(self.device)
                if self.opt.gp_weights == 'uniform':
                    self.gp_weights = [1.0] * 6
                else:
                    self.gp_weights = eval(self.opt.gp_weights)
                self.loss_names += ['GP']

            if self.opt.lambda_asp > 0:
                self.criterionASP = AdaptiveSupervisedPatchNCELoss(self.opt).to(self.device)
                self.loss_names += ['ASP']

            if self.opt.lambda_consist > 0:
                self.criterionConsist = torch.nn.L1Loss().to(self.device)
                self.loss_names += ['consist']

            if self.opt.lambda_LSH > 0:
                assert self.opt.LSH_type in ['all', 'weak_paired']
                self.loss_names += ['LSH']
                self.LSH_scalers = [float(t) for t in self.opt.LSH_scalers.split(',')]
                self.LSH_clusters = [None for _ in range(len(self.LSH_scalers))]


    def data_dependent_initialize(self, data, total_iters=-1):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        bs_per_gpu = data["A"].size(0) // max(len(self.opt.gpu_ids), 1)
        self.set_input(data, total_iters)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.forward()                                          # compute fake images: G(A)
        if self.opt.isTrain:
            self.compute_D_loss().backward()                  # calculate gradients for D
            self.compute_G_loss().backward()                   # calculate graidents for G
            if self.opt.lambda_NCE > 0.0 or self.opt.lambda_asp > 0.0:
                self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
                self.optimizers.append(self.optimizer_F)

    def optimize_parameters(self):
        # forward
        self.forward()

        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()
        self.optimizer_D.step()
        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.step()

   
    def set_input(self, input, total_iters=-1):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        self.isConsistent = input['isConsistent'].item()
        if total_iters == 1:
            self.isConsistent = True

        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

        if self.opt.use_semistain:
            self.aug_real_A = input['A_aug'].to(self.device) 

        if 'current_epoch' in input:
            self.current_epoch = input['current_epoch']
        if 'current_iter' in input:
            self.current_iter = input['current_iter']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.real = torch.cat((self.real_A, self.real_B), dim=0) if self.opt.nce_idt and self.opt.isTrain else self.real_A
        if self.opt.flip_equivariance:
            self.flipped_for_equivariance = self.opt.isTrain and (np.random.random() < 0.5)
            if self.flipped_for_equivariance:
                self.real = torch.flip(self.real, [3])

        if not self.opt.use_semistain or self.isConsistent or self.opt.phase == 'test':
            if 'G' in self.model_names:
                self.fake = self.netG(self.real, layers=[])
            elif 'G_EMA' in self.model_names:
                self.fake = self.netG_EMA(self.real, layers=[])
            else:
                raise NotImplementedError

            self.fake_B = self.fake[:self.real_A.size(0)]
            if self.opt.nce_idt:
                self.idt_B = self.fake[self.real_A.size(0):]

         

    def EMA_update(self, model, model_EMA):
        model_state_dict = model.state_dict()
        model_EMA_state_dict = model_EMA.state_dict()

        for name in model_EMA_state_dict: 
            assert name in model_state_dict
            model_EMA_state_dict[name].data = model_EMA_state_dict[name].data * self.opt.ema + model_state_dict[name].data * (1 - self.opt.ema)

        model_EMA.load_state_dict(model_EMA_state_dict)

    def compute_D_loss(self):
        """Calculate GAN loss for the discriminator"""
        if not self.opt.use_semistain or self.isConsistent or self.opt.phase == 'test':
            fake = self.fake_B.detach()
        else:
            fake = self.netG(self.real_A, layers=[]).detach()
        pred_fake = self.netD(fake)
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
        self.pred_real = self.netD(self.real_B)
        loss_D_real = self.criterionGAN(self.pred_real, True)
        self.loss_D_real = loss_D_real.mean()

        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D

    def compute_G_loss(self):
        """Calculate GAN and NCE loss for the generator"""
        assert self.opt.phase == 'train'

        if not self.opt.use_semistain or self.isConsistent or self.opt.phase == 'test':
            fake = self.fake_B

            feat_real_A = self.netG(self.real_A, self.nce_layers, encode_only=True)
            feat_fake_B = self.netG(self.fake_B, self.nce_layers, encode_only=True)
            feat_real_B = self.netG(self.real_B, self.nce_layers, encode_only=True)
            
            if self.opt.nce_idt:
                feat_idt_B = self.netG(self.idt_B, self.nce_layers, encode_only=True)

            # First, G(A) should fake the discriminator
            if self.opt.lambda_GAN > 0.0:
                pred_fake = self.netD(fake)
                self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
            else:
                self.loss_G_GAN = 0.0

            if self.opt.lambda_NCE > 0.0:
                self.loss_NCE = self.calculate_NCE_loss(feat_real_A, feat_fake_B, self.netF, self.nce_layers)
            else:
                self.loss_NCE, self.loss_NCE_bd = 0.0, 0.0
        
            if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:
                self.loss_NCE_Y = self.calculate_NCE_loss(feat_real_B, feat_idt_B, self.netF, self.nce_layers)
            else:
                self.loss_NCE_Y = 0.0

            # FDL: NCE between the noisy pairs (fake_B and real_B)
            if self.opt.lambda_asp > 0:
                self.loss_ASP = self.calculate_NCE_loss(feat_real_B, feat_fake_B, self.netF, self.nce_layers, paired=True)
            else:
                self.loss_ASP = 0.0

            self.loss_consist = 0.0

            # FDL: compute loss on Gaussian pyramids
            if self.opt.lambda_gp > 0:
                p_fake_B = self.P(self.fake_B)
                p_real_B = self.P(self.real_B)
                loss_pyramid = [self.criterionGP(pf, pr) for pf, pr in zip(p_fake_B, p_real_B)]
                weights = self.gp_weights
                loss_pyramid = [l * w for l, w in zip(loss_pyramid, weights)]
                self.loss_GP = torch.mean(torch.stack(loss_pyramid)) * self.opt.lambda_gp
            else:
                self.loss_GP = 0

            if self.opt.lambda_LSH > 0 and self.opt.LSH_type == 'all':
                self.loss_LSH = self.calculate_LSH_loss(feat_real_A, feat_fake_B, self.netF, self.nce_layers) * self.opt.lambda_LSH
            else:
                self.loss_LSH = 0

            self.loss_G = self.loss_G_GAN + self.loss_NCE + self.loss_NCE_Y + self.loss_ASP + self.loss_GP + self.loss_LSH
            if self.opt.use_semistain:
                self.EMA_update(self.netG, self.netG_EMA)

        else:
            aug_fake_B = self.netG(self.aug_real_A, layers=[])
            ema_fake_B = self.netG_EMA(self.real_A, layers=[])

            if self.opt.lambda_GAN > 0.0:
                pred_fake = self.netD(aug_fake_B)
                self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
            else:
                self.loss_G_GAN = 0.0

            feat_real_A = self.netG(self.real_A, self.nce_layers, encode_only=True)
            feat_aug_fake_B = self.netG(aug_fake_B, self.nce_layers, encode_only=True)

            if self.opt.lambda_NCE > 0.0:
                self.loss_NCE = self.calculate_NCE_loss(feat_real_A, feat_aug_fake_B, self.netF, self.nce_layers)
            else:
                self.loss_NCE, self.loss_NCE_bd = 0.0, 0.0
        
            self.loss_consist = self.criterionConsist(aug_fake_B, ema_fake_B.detach()) * self.opt.lambda_consist

            if self.opt.lambda_LSH > 0:
                self.loss_LSH = self.calculate_LSH_loss(feat_real_A, feat_aug_fake_B, self.netF, self.nce_layers) * self.opt.lambda_LSH
            else:
                self.loss_LSH = 0

            self.loss_G = self.loss_G_GAN + self.loss_NCE + self.loss_consist + self.loss_LSH

            self.loss_NCE_Y = 0.0
            self.loss_ASP = 0.0 
            self.loss_GP = 0.0            

        return self.loss_G

    def calculate_NCE_loss(self, feat_src, feat_tgt, netF, nce_layers, paired=False):
        n_layers = len(feat_src)
        feat_q = feat_tgt

        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]
        feat_k = feat_src
        feat_k_pool, sample_ids = netF(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = netF(feat_q, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k in zip(feat_q_pool, feat_k_pool):
            if paired:
                loss = self.criterionASP(f_q, f_k, self.current_epoch) * self.opt.lambda_asp
            else:
                loss = self.criterionNCE(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers

    def merge_cluster(self, coarse_cluster_labels, threshold=0.01):
        coarse_cluster_values, per_coarse_culsters_num = torch.unique(coarse_cluster_labels, return_counts=True)
        for i in range(coarse_cluster_values.shape[0]):
            label, num = coarse_cluster_values[i], per_coarse_culsters_num[i]
            if num / torch.sum(per_coarse_culsters_num) < threshold:
                coarse_cluster_labels[coarse_cluster_labels.squeeze() == label] = -1
        return coarse_cluster_labels

    def calculate_LSH_loss(self, feat_src, feat_tgt, netF, nce_layers, paired=False, visualize=False):
        n_layers = len(feat_src)
        feat_q = feat_tgt 
        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]
        feat_k = feat_src 
       
        assert self.LSH_scalers[3] == 0 and self.LSH_scalers[4] == 0
        feat_k_pool, sample_ids_k = netF(feat_k[:3], self.opt.LSH_patch_num, None, random_permute=True, sub_patch_size=3)
        feat_q_pool, _ = netF(feat_q[:3], self.opt.LSH_patch_num, sample_ids_k, random_permute=True, sub_patch_size=3)

        total_lch_loss = 0.0
        for i, (f_q, f_k) in enumerate(zip(feat_q_pool, feat_k_pool)):
            if self.LSH_scalers[i] == 0:
                continue

            if self.opt.fixed_cluster == 'True':

                if self.LSH_clusters[i] == None:
                    Cluster = MultiLSHRandomProjection(d_model=f_k.shape[1], b=self.opt.LSH_ncluster, device=f_k.device, num_tables=self.opt.num_tables, dis_threshold=1)
                    self.LSH_clusters[i] = Cluster
                else:
                    Cluster = self.LSH_clusters[i]
            else:
                Cluster = MultiLSHRandomProjection(d_model=f_k.shape[1], b=self.opt.LSH_ncluster, device=f_k.device, num_tables=self.opt.num_tables, dis_threshold=1)

            cluster_labels = Cluster.cluster(f_k.detach())
            cluster_labels = self.merge_cluster(cluster_labels)
            cluster_values, per_culsters_num = torch.unique(cluster_labels, return_counts=True)
            
            H, W = feat_k[i].shape[2:]

            if visualize:    
                coord_x_A, coord_y_A = sample_ids_k[i] // H, sample_ids_k[i] % H 
                cluster_mask = torch.ones(H, W) * -2
                for j in range(sample_ids_k[i].shape[0]):
                    cluster_mask[coord_x_A[j], coord_y_A[j]] = cluster_labels[j]
                cluster_mask = (cluster_mask - torch.min(cluster_mask)) / (torch.max(cluster_mask) - torch.min(cluster_mask)) * 255.
                cluster_mask = cv2.applyColorMap(np.array(cluster_mask).astype(np.uint8), 2)
                cv2.imwrite(f'visualize/cluster_{i}_sps=3.jpg', cluster_mask[:,:,(2,1,0)])

            num_clusters = cluster_values.shape[0]

            if num_clusters > 1:    
                simialrity_exp = torch.exp(torch.matmul(f_q, f_q.T) / self.opt.nce_LSH)

                same_cluster = (cluster_labels.view(-1, 1) == cluster_labels.view(1, -1)).float().to(f_q.device)
                pos_mask = same_cluster.fill_diagonal_(0)
                neg_mask = (1 - same_cluster).fill_diagonal_(0)
                positive_sims = simialrity_exp * pos_mask
                negative_sims = simialrity_exp * neg_mask
                infonces = -torch.log(torch.sum(positive_sims, 1) / (torch.sum(positive_sims, 1) + torch.sum(negative_sims, 1)) + 1e-16)

                
                total_lch_loss += torch.mean(infonces) * self.LSH_scalers[i]

        return total_lch_loss / torch.sum(torch.tensor(self.LSH_scalers) > 0)

    
