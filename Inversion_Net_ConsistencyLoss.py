from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import random
import torch
import torch.nn as nn
# import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
# import torch.utils.data
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils

import math
import numbers
import numpy as np
import pandas as pd
import os
import copy
import glob
import collections
from PIL import Image
#import cv2
import sys 
import matplotlib.pyplot as plt

sys.path.insert(0,os.path.abspath('..'))

from resnet_cifar_content import ResNet34,ResNet50, ResNet18
from utils import get_cw_VR_loss, InversionActivationHook
from utils import rs_maker, denormalize, ZipCreator, Cutout
from ConsistencyLoss_utils import NetWrapper, MLP, EMA

NUM_CLASSES = 10
ALPHA=1.0
image_list=[]
target_list=[]
#os.environ["CUDA_VISIBLE_DEVICES"] = args.ngpu
try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex import amp, optimizers
    USE_APEX = True
except ImportError:
    print("Please install apex from https://www.github.com/nvidia/apex to run this example.")
    print("will attempt to run without it")
    USE_APEX = False

#provide intermeiate information
debug_output = False
debug_output = True

def gram(x):
	b,c,h,w = x.size()
	x = x.view(b*c, -1)
	return torch.mm(x, x.t())

def gram_no(x):
	c,h,w = x.size()
	x = x.view(c, -1)
	return torch.mm(x, x.t())

# To fix Seed
def random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

def cov(tensor, rowvar=True, bias=False):
    """Estimate a covariance matrix (np.cov)"""
    tensor = tensor if rowvar else tensor.transpose(-1, -2)
    tensor = tensor - tensor.mean(dim=-1, keepdim=True)
    factor = 1 / (tensor.shape[-1] - int(not bool(bias)))
    return factor * tensor @ tensor.transpose(-1, -2).conj()

class DeepInversionFeatureHook():
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''

    def __init__(self, module, rs):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.rs = rs

    def hook_fn(self, module, input, output):
        # hook co compute deepinversion's feature distribution regularization
        bs = input[0].shape[0]
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)
        
        r_feature = torch.norm(module.running_var.data.type(var.type()) - var, 1) + torch.norm(
            module.running_mean.data.type(var.type())  - mean, 1)

        self.r_feature = r_feature

    def close(self):
        self.hook.remove()


def smoothing_onehot(labels):
    one_hot = F.one_hot(labels, NUM_CLASSES)
    target_ = ALPHA * one_hot
    for i in range(args.bs):
        random_num = list(range(0,NUM_CLASSES))
        random_num.remove(labels[i])
        ri=random.choice(random_num)
        target_[i][ri]=1-ALPHA
    return target_

def kl_loss2(x_target,x_pred):
    eps=1e-7
    x_target = x_target+eps
    assert x_target.size() == x_pred.size(),"size fail ! "+str(x_target.size())+" "+str(x_pred.size())
    logged_x_pred = torch.log(x_pred)
    logged_x_target = torch.log(x_target)
    cost_value = (torch.sum(x_target*logged_x_target)-(torch.sum(x_target*logged_x_pred)))/x_target.size(0)
    return cost_value

def make_grid(tensor, nrow, padding = 2, pad_value : int = 0):
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    num_channels = tensor.size(1)
    grid = tensor.new_full((num_channels, height * ymaps + padding, width * xmaps + padding), pad_value)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            # Tensor.copy_() is a valid method but seems to be missing from the stubs
            # https://pytorch.org/docs/stable/tensors.html#torch.Tensor.copy_
            grid.narrow(1, y * height + padding, height - padding).narrow(  # type: ignore[attr-defined]
                2, x * width + padding, width - padding
            ).copy_(tensor[k])
            k = k + 1
    return grid

def consistency_loss(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)


def get_images(net, bs=128, epochs=1000, idx=-1, var_scale=0.00005,
               net_student=None, prefix=None, competitive_scale=0.01, train_writer = None, global_iteration=None,
               use_amp=False, optimizers = None, rs = 0, bn_reg_scale = 0.0, random_labels = False, l2_coeff=0.0, 
               generators = None, version = None, online = None, target = None, ema_updater = None, G_target = None):
    '''
    Function returns inverted images from the pretrained model, parameters are tight to CIFAR dataset
    args in:
        net: network to be inverted
        bs: batch size
        epochs: total number of iterations to generate inverted images, training longer helps a lot!
        idx: an external flag for printing purposes: only print in the first round, set as -1 to disable
        var_scale: the scaling factor for variance loss regularization. this may vary depending on bs
            larger - more blurred but less noise
        net_student: model to be used for Adaptive DeepInversion
        prefix: defines the path to store images
        competitive_scale: coefficient for Adaptive DeepInversion
        train_writer: tensorboardX object to store intermediate losses
        global_iteration: indexer to be used for tensorboard
        use_amp: boolean to indicate usage of APEX AMP for FP16 calculations - twice faster and less memory on TensorCores
        optimizer: potimizer to be used for model inversion
        inputs: data place holder for optimization, will be reinitialized to noise
        bn_reg_scale: weight for r_feature_regularization
        random_labels: sample labels from random distribution or use columns of the same class
        l2_coeff: coefficient for L2 loss on input
    return:
        A tensor on GPU with shape (bs, 3, 32, 32) for CIFAR
    '''

    online_classifier, online_projector, online_predictor, optimizer_online = online
    target_classfier, target_projector = target

    kl_loss = nn.KLDivLoss(reduction='batchmean').cuda()

    net_student.eval()

    best_cost = 1e6

    optimizer = optimizers
    G = generators
    G_target = G_target


    # Learnable Scale Parameter
    
    alpha = 3*torch.ones((bs, 3, 1, 1), device = device)
    alpha.requires_grad = True
    optimizer_alpha = torch.optim.Adam([alpha], lr = 0.01)

    # if use_amp:
    #     inputs.data = inputs.data.half()

    # set up criteria for optimization
    criterion = nn.CrossEntropyLoss()
    #se_loss = torch.nn.MSELoss()
    optimizer.state = collections.defaultdict(dict)
    optimizer_alpha.state = collections.defaultdict(dict)

    # Concat Z
    
    #targets = torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * 12 + [random.randint(0,9) for i in range(8)]).to('cuda')
    #print(targets)
    #targets = torch.LongTensor([i for i in range(100)] + [j for j in range(28)]).to('cuda')
    #targets = torch.LongTensor([i for i in range(100)]*2 + [j for j in range(56)]).to('cuda')
    #targets = torch.LongTensor(torch.randint(0, 10, (bs,))).to(device)
    #print(targets)
    targets = torch.LongTensor([0] * 12 + [1] * 12 + [2] * 12 + [3] * 12 + [4] * 12 + [5] * 12 + [6] * 12 + [7] * 12 + [8] * 12 + [9] * 12).to('cuda')
    #targets = torch.LongTensor([0, 1, 2, 3, 4]).to('cuda')
    #targets = torch.LongTensor([1]*bs).to('cuda')
    tf_targets = smoothing_onehot(targets)
    #z = torch.randn((args.bs, 128)).to(device)
    #z = torch.cat((z,tf_targets), dim = 1)
    #print('Z shape:',z.shape)

    ## Create hooks for feature statistics catching
    loss_r_feature_layers = []
    count = 0
    
    for module in online_classifier.modules():
        if isinstance(module, nn.BatchNorm2d):
            loss_r_feature_layers.append(DeepInversionFeatureHook(module, 0))
    
    #smoothing = GaussianSmoothing(3, 5, 3).to(device)
    # setting up the range for jitter
    lim_0, lim_1 = 2, 2
    target_list = []
    bn_list = []
    for epoch in range(epochs):
        cutout = Cutout(n_holes = 1, length = 18, gpu = 0)
        
        #targets = torch.LongTensor(torch.randint(0, 10, (bs,))).to(device)
        #tf_targets = smoothing_onehot(targets)
        z1 = torch.randn((args.bs, 128)).to(device)
        z2 = torch.randn((args.bs, 128)).to(device)
        
        z1 = torch.cat((z1, tf_targets), dim = 1)
        z2 = torch.cat((z2, tf_targets), dim = 1)

        nu = random.randint(0, 119)
        alist=[]                          # 뽑은 a를 넣어 중복 방지해주는 리스트         
        for i in range(nu):
            a = random.randint(0,119)       
            while a in alist :              # a가 이미 뽑은 리스트에 있을 때까지 다시 뽑자
                a = random.randint(0,119)
            alist.append(a)
        se = np.array(alist)
        #crop = 0

        #z = torch.cat((z,tf_targets), dim = 1)
        inputs_jit1 = G(z1)
        inputs_jit2 = G(z2)

        inputs_jit1 = alpha * inputs_jit1
        inputs_jit2 = alpha * inputs_jit2

        # apply random jitter offsets
        off1 = random.randint(-lim_0, lim_0)
        off2 = random.randint(-lim_1, lim_1)
        inputs_jit1 = torch.roll(inputs_jit1, shifts=(off1,off2), dims=(2,3))
        off1 = random.randint(-lim_0, lim_0)
        off2 = random.randint(-lim_1, lim_1)
        inputs_jit2 = torch.roll(inputs_jit2, shifts=(off1,off2), dims=(2,3))
        
        # Apply random flip 
        flip = random.random() > 0.8
        if flip:
            inputs_jit1 = torch.flip(inputs_jit1, dims = (3,))
        flip = random.random() > 0.8
        if flip:
            inputs_jit2 = torch.flip(inputs_jit2, dims = (3,))

        crop = random.random() >= 0.8
        if crop:
            inputs_jit1[se] = cutout(inputs_jit1[se])
        crop = random.random() >= 0.8
        if crop:
            inputs_jit2[se] = cutout(inputs_jit2[se])
        #outputs = net(inputs_jit)

        outputs1, representation1 = online_classifier(inputs_jit1)
        #print(outputs1[0].shape)
        #print(online_classifier)
        online_proj_1 = online_projector(representation1)
        online_pred_1 = online_predictor(online_proj_1)
        outputs2, representation2 = online_classifier(inputs_jit2)
        online_proj_2 = online_projector(representation2)
        online_pred_2 = online_predictor(online_proj_2)

        ## Input2
        with torch.no_grad():
            t_inputs_jit1 = G_target(z1)
            t_inputs_jit2 = G_target(z2)
            # apply random jitter offsets
            off1 = random.randint(-lim_0, lim_0)
            off2 = random.randint(-lim_1, lim_1)
            t_inputs_jit1 = torch.roll(t_inputs_jit1, shifts=(off1,off2), dims=(2,3))
            off1 = random.randint(-lim_0, lim_0)
            off2 = random.randint(-lim_1, lim_1)
            t_inputs_jit2 = torch.roll(t_inputs_jit2, shifts=(off1,off2), dims=(2,3))
            
            # Apply random flip 
            flip = random.random() > 0.8
            if flip:
                t_inputs_jit1 = torch.flip(t_inputs_jit1, dims = (3,))
            flip = random.random() > 0.8
            if flip:
                t_inputs_jit2 = torch.flip(t_inputs_jit2, dims = (3,))
                
            crop = random.random() >= 0.8
            if crop:
                t_inputs_jit1[se] = cutout(t_inputs_jit1[se])
            crop = random.random() >= 0.8
            if crop:
                t_inputs_jit2[se] = cutout(t_inputs_jit2[se])

            t_output1, t_representation1 = target_classifier(t_inputs_jit1)
            target_proj_1 = target_projector(t_representation1)
            t_output2, t_representation2 = target_classifier(t_inputs_jit2)
            target_proj_2 = target_projector(t_representation2)

        optimizer.zero_grad()
        #optimizer_f.zero_grad()
        optimizer_alpha.zero_grad()
        optimizer_online.zero_grad()
        #optimizer_beta.zero_grad()

        #print(outputs)
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        #me = inputs_jit.mean(dim = (0, 2, 3))
        #va = inputs_jit.var(dim = (0, 2, 3))

        #input_loss = mse_loss(me, torch.tensor([0.4914, 0.4822, 0.4465]).to('cuda')) + mse_loss(va, torch.tensor([0.2023, 0.1994, 0.2010]).to('cuda'))
        loss_target = criterion(outputs1[0], targets)
        loss = loss_target

        # apply total variation regularization
        diff1 = inputs_jit1[:,:,:,:-1] - inputs_jit1[:,:,:,1:]
        diff2 = inputs_jit1[:,:,:-1,:] - inputs_jit1[:,:,1:,:]
        diff3 = inputs_jit1[:,:,1:,:-1] - inputs_jit1[:,:,:-1,1:]
        diff4 = inputs_jit1[:,:,:-1,:-1] - inputs_jit1[:,:,1:,1:]
        loss_var = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
        loss = loss + var_scale*loss_var

        # R_feature loss
        if 1:
            loss_distr = sum([mod.r_feature for idx, mod in enumerate(loss_r_feature_layers)])
            loss = loss + bn_reg_scale * loss_distr

        # l2 loss
        if 1:
            loss = loss + l2_coeff * torch.norm(inputs_jit1, 2)
        
        if 1:
            alpha_tv = torch.sum(torch.var(alpha, dim = 1, unbiased = True))
            loss = loss + var_scale * alpha_tv

        # Consistency Loss
        if 1:
            loss_one = consistency_loss(online_pred_1, target_proj_2.detach())
            loss_two = consistency_loss(online_pred_2, target_proj_1.detach())
            loss_con = (loss_one + loss_two).mean()
            loss += 0.01 * loss_con # BYOL
            #loss -= 0.01 * loss_con # RAFT


        if debug_output and epoch % 100==0:

            #df = np.array(target_list)
            #pd.DataFrame(df).to_csv('./{}/target_loss.csv'.format(prefix))
            #df = np.array(bn_list)
            #pd.DataFrame(df).to_csv('./{}/bn_loss.csv'.format(prefix))
            #print('targets :', targets)
            #print('alpha:', alpha)
            print("It {}\t Losses: total: {:.3f},\ttarget: {:.3f} \tR_feature_loss unscaled:\t {:.3f}\t grad_loss : {:.3f} \t gram_loss:{:.3f}".format(epoch, 
                    loss.item(),loss_target.item(),loss_distr.item(), loss_con.item(),0))
            #print('fc1:{:.4f}, fc2:{:.4f}, fc3:{:.4f}, fc4{:.4f}, fc5{:.4f}'.format(mseloss1,mseloss2,mseloss3,mseloss4,mseloss5))
            #print('fc1 between class:{:.4f}, fc2:{:.4f}, fc3:{:.4f}, fc4:{:.4f}, fc5{:.4f}'.format(mseloss_between_class1,mseloss_between_class2,mseloss_between_class3
            #    ,mseloss_between_class4, mseloss_between_class5))
            #print(inputs_z.data.clone()[0])
            nchs = inputs_jit1.shape[1]
            #print('mean',inputs_z.data.clone().mean([0,2,3]))
            #print('var', inputs_z.data.clone().permute(1,0,2,3).contiguous().view(nchs, -1).var(1, unbiased = False))
            image_tensor = denormalize(inputs_jit1.data.clone())
            grid = make_grid(image_tensor, nrow = 10)
            ndarr = grid.mul(255).permute(1,2,0).to('cpu', torch.uint8).numpy()
            im = Image.fromarray(ndarr)
            #im.save('./{}/denorm_{}.png'.format(prefix, epoch//100))
            vutils.save_image(inputs_jit1.data.clone(),
                            './{}/output_{}.png'.format(prefix, epoch//100),
                            normalize=True, scale_each=True, nrow=10)
            vutils.save_image(inputs_jit2.data.clone(),
                            './{}/2output_{}.png'.format(prefix, epoch//100),
                            normalize=True, scale_each=True, nrow=10)
            
        if best_cost > loss.item():
            best_cost = loss.item()
            with torch.no_grad():
                inputs_for_save = alpha * G(z1)
            best_inputs = inputs_for_save
            best_G = G.state_dict()

        # backward pass
        if use_amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        optimizer.step()
        optimizer_alpha.step()
        optimizer_online.step()
        ema_updater.update(online_projector, target_projector)
        ema_updater.update(G, G_target)

    #torch.save(best_G, './data_weight/DI_GEN_upsample_3x3.pt')

    #outputs=net(best_inputs)
    #_, predicted_teach = outputs.max(1)

    #outputs_student=net_student(best_inputs)
    #_, predicted_std = outputs_student.max(1)

    #if idx == 0:
    #    print('Teacher correct out of {}: {}, loss at {}'.format(bs, predicted_teach.eq(targets).sum().item(), criterion(outputs, targets).item()))
    #    print('Student correct out of {}: {}, loss at {}'.format(bs, predicted_std.eq(targets).sum().item(), criterion(outputs_student, targets).item()))

    name_use = "best_images"
    if prefix is not None:
        name_use = prefix + name_use
    next_batch = len(glob.glob("./%s/*.png" % name_use)) // 1

    vutils.save_image(best_inputs[:20].clone(),
                      './{}/output_{}.png'.format(name_use, next_batch),
                      normalize=True, scale_each = True, nrow=10)

    if train_writer is not None:
        train_writer.add_scalar('gener_teacher_criteria', criterion(outputs, targets), global_iteration)
        train_writer.add_scalar('gener_student_criteria', criterion(outputs_student, targets), global_iteration)

        train_writer.add_scalar('gener_teacher_acc', predicted_teach.eq(targets).sum().item() / bs, global_iteration)
        train_writer.add_scalar('gener_student_acc', predicted_std.eq(targets).sum().item() / bs, global_iteration)

        train_writer.add_scalar('gener_loss_total', loss.item(), global_iteration)
        train_writer.add_scalar('gener_loss_var', (var_scale*loss_var).item(), global_iteration)

    net_student.train()
    #torch.save(G.state_dict(), './data_weight/Z_mv_generator2.pth')
    #torch.save(z, './data_weight/z2.pth')

    return best_inputs,targets

def save_finalimages(images, targets, num_generations, prefix, exp_descr):
    # method to store generated images locally
    local_rank = torch.cuda.current_device()
    
    for id in range(images.shape[0]):
        class_id = targets[id].item()
        image = images[id].reshape(3,32,32)
        image_np = images[id].data.cpu().numpy()
        pil_image = torch.from_numpy(image_np)
        '''
        save_pth = os.path.join(prefix, args.final_data_path + '/s{}/{}_output_{}_3.png'.format(class_id, num_generations, id))
        if not os.path.exists(save_pth):
            os.makedirs(save_pth)
        '''
        image_list.append(pil_image)
        target_list.append(class_id)

        if 0:
            #save into separate folders
            place_to_store = '{}/s{}/img_{:05d}_id{:03d}_gpu_{}_2.jpg'.format(args.final_data_path, class_id,
                                                                                       num_generations, id,
                                                                                          local_rank)
        else:
            place_to_store = '{}/img_s{:03d}_{:05d}_id{:03d}_gpu_{}_2.jpg'.format(args.final_data_path, class_id,
                                                                                          num_generations, id,
                                                                                          local_rank)

        vutils.save_image(image, os.path.join(prefix, args.final_data_path + '/s{}/{}_output_{}_'.format(class_id, num_generations, id)) + exp_descr + '.png',
                              normalize=True, scale_each=True, nrow=1)
    loader = {'data':image_list,'target':target_list}


def test(net):
    print('==> Teacher validation')
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    print('Loss: %.3f | Acc: %.3f%% (%d/%d)'
          % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Inversion')
    parser.add_argument('--ngpu', type=str, default='2', metavar='strN',help='device number as a string') 
    parser.add_argument('--bs', default=128, type=int, help='batch size')
    parser.add_argument('--iters_mi', default=2000, type=int, help='number of iterations for model inversion')
    parser.add_argument('--cig_scale', default=0.0, type=float, help='competition score')
    parser.add_argument('--di_lr', default=0.0001, type=float, help='lr for deep inversion')
    parser.add_argument('--D_lr', default=0.00001, type=float, help='lr for deep inversion')
    parser.add_argument('--di_var_scale', default=6.0e-3, type=float, help='TV L2 regularization coefficient')
    parser.add_argument('--di_l2_scale', default=1.5e-5, type=float, help='L2 regularization coefficient')
    parser.add_argument('--r_feature_weight', default=5.0, type=float, help='weight for BN regularization statistic')
    parser.add_argument('--amp', action='store_true', help='use APEX AMP O1 acceleration')
    parser.add_argument('--exp_descr', default="try1", type=str, help='name to be added to experiment name')
    parser.add_argument('--teacher_weights', default='./pretrained/resnet34.pt', type=str, help='path to load weights of the model')
    parser.add_argument('--final_data_path', default='sample_image',type=str,help='path to save final inversion images')
    parser.add_argument('--version', default = 1, type = int, help = 'Choose version for Generator')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.ngpu
    global_iteration=79

    print("loading resnet34")
    net_teacher = ResNet34(num_classes = 10)
    net_student = ResNet18()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net_student = net_student.to(device)
    net_teacher = net_teacher.to(device)
    criterion = nn.CrossEntropyLoss()

    checkpoint = torch.load(args.teacher_weights)
    net_teacher.load_state_dict(checkpoint)
    net_teacher.eval() #important, otherwise generated images will be non natural
    
    # Fix Seed
    #if 1:
    #    random_seed(57) 

    # Select Version for Generator
    if args.version == 1:
        from generator_ver2 import Generator
        G_online = Generator(8, 138, 3).to(device)
    
    G_target = copy.deepcopy(G_online)
    for p in G_target.parameters():
        p.requires_grad = False
    
    #optimizer_di = optim.SGD(G_online.parameters(), lr = args.di_lr, momentum = 0.9, nesterov = True)
    optimizer_di = optim.Adam(G_online.parameters(), lr = args.di_lr)

    ## Wrap the online Classfier
    online_classifier = NetWrapper(net_teacher, 4096, 256)
    online_classifier.eval()
    
    online_projector = MLP(512, 4096, 256).to(device)
    online_predictor = MLP(256, 4096, 256).to(device)
    optimizer_online = optim.Adam(list(online_projector.parameters())+list(online_predictor.parameters()), lr = args.di_lr)
    #optimizer_online2 = optim.Adam([online_projector.parameters(), online_predictor.parameters()], lr = args.di_lr)

    target_classifier = copy.deepcopy(online_classifier)
    for p in target_classifier.parameters():
        p.requires_grad = False
    target_projector = copy.deepcopy(online_projector)
    for p in online_projector.parameters():
        p.requires_grad = False
    ema_updater = EMA(0.99)


    for i in range(global_iteration):
        print("iteration number : ",i)
        # place holder for inputs
        data_type = torch.half if args.amp else torch.float

        if args.amp:
            opt_level = "O1"
            loss_scale = 'dynamic'

            [net_student, net_teacher], optimizer_di = amp.initialize(
             [net_student, net_teacher], optimizer_di,
             opt_level=opt_level,
             loss_scale=loss_scale)
        
        if args.amp:
            print('hi')
            # need to do this trick for FP16 support of batchnorms
            net_teacher.train()
            for module in net_teacher.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval().half()
        cudnn.benchmark = True
        batch_idx = 0
        
        prefix_ = "image/CIFAR10_consistency_loss/data_generation"
        prefix = os.path.join(prefix_,args.exp_descr+str(i)+"/")

        for create_folder in [prefix, prefix+"/best_images/"]:
            if not os.path.exists(create_folder):
                os.makedirs(create_folder)



        print("Starting model inversion")
        inputs,targets = get_images(net=net_teacher, bs=args.bs, epochs=args.iters_mi, idx=batch_idx,
                        net_student=net_student, prefix=prefix, competitive_scale=args.cig_scale,
                        train_writer=None, global_iteration=global_iteration, use_amp=args.amp,
                        optimizers= optimizer_di, rs = 0, bn_reg_scale=args.r_feature_weight,
                        var_scale=args.di_var_scale, random_labels=False, l2_coeff=args.di_l2_scale, generators = G_online, 
                        version = args.version, online = (online_classifier, online_projector, online_predictor, optimizer_online),
                        target = (target_classifier, target_projector), ema_updater = ema_updater,
                        G_target = G_target)
        torch.save(inputs,os.path.join(prefix,'data' + str(i)))
        save_finalimages(inputs,targets,i, prefix_, args.exp_descr)
