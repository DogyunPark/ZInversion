import torch
import sys
import torch.nn as nn
#from batchinstancenorm import BatchInstanceNorm2d
import os
sys.path.insert(0, os.path.abspath('..'))
#import biggan.layers as layers

def normalization(x):
    bs = x.shape[0]
    ch = x.shape[1]
    mean = x.mean(dim = (2,3), keepdim = True)
    var = x.var(dim = (2,3), unbiased = False, keepdim = True)
    x = (x - mean) / torch.sqrt(var)
    #x = (x - mean)
    
    return x

class Generator(nn.Module):
    
    def __init__(self, image_size, latent_dim, channel, init = False, F_init = 'ortho'):
        super(Generator, self).__init__()
        self.init_size = image_size
        self.F_init = F_init
        self.l1 = nn.Sequntial(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace = True)
        )
        self.l2 = nn.Sequential(
            nn.Linear(hidden_dim, project_dim),
            nn.BatchNomr1d(project_dim),
            nn.ReLU(inplace = True)
        )
        #self.l2 = nn.Linear(latent_dim, 512)
        self.l3 = nn.Linear(latent_dim, 128 * image_size ** 2)
        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(128),
            #BatchInstanceNorm2d(128),
            #layers.SandwichBatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.Tanh(),
            #nn.Sigmoid(),
            #nn.ReLU(True),
            nn.Upsample(scale_factor = 2),
            nn.Conv2d(128, 128, 3, stride = 1, padding = 1),
            #nn.ConvTranspose2d(128,128,kernel_size = 3, stride = 2, padding = 1, output_padding = 1),
            nn.BatchNorm2d(128),
            #BatchInstanceNorm2d(128),
            #layers.SandwichBatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace = True),
            #nn.Tanh(),
            #nn.Sigmoid(),
            #nn.ReLU(True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride = 1, padding = 1),
            #nn.ConvTranspose2d(128, 64, 3, stride= 2, padding= 1, output_padding = 1),
            nn.BatchNorm2d(64),
            #BatchInstanceNorm2d(64),
            #layers.SandwichBatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace = True),
            #nn.Tanh(),
            #nn.Sigmoid(),
            #nn.ReLU(True),
            #nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 3, 3, stride = 1, padding = 1),
            #nn.LeakyReLU(0.2, inplace = True),
            #nn.Tanh(),
            #nn.BatchNorm2d(3),
            #nn.Tanh(),
            #nn.BatchNorm2d(3),
            #nn.Tanh()
            #nn.ConvTranspose2d(64, 3, 3, stride= 1, padding= 1),
        )
        #self.l2 = nn.Linear(32*32*3, latent_dim)
        #self.relu = nn.ReLU(inplace=True)

        if init:
            self.init_weights()

    def init_weights(self):
        self.nu = 0
        self.param_count = 0
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) or isinstance(module, nn.Embedding):
                if self.F_init == 'ortho':
                    nn.init.orthogonal_(module.weight)
                elif self.F_init == 'normal':
                    nn.init.normal_(module.weight, 0, 0.02)
                elif self.F_init == 'xavier':
                    nn.init.xavier_uniform_(module.weight)
                else:
                    raise ValueError('Invalid Initialization Method')
                self.param_count += sum([p.data.nelement() for p in module.parameters()])
            elif isinstance(module, nn.BatchNorm2d):
                self.nu += 1
        print('Param Count for FeatureDecoder initialized parameters : {:d}'.format(self.param_count))
        print('BatchNorm Layer : {:d}'.format(self.nu))

    def forward(self, z):
        z = self.l1(z)
        z = self.l2(z)
        out1 = self.l3(z)
        out = out1.view(out1.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_block(out)
        #print(img.shape)
        img = nn.functional.tanh(img)
        #z_out = self.l2(out.reshape(-1,32*32*3))
        return img 

class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            #nn.BatchNorm2d(16),
            nn.ReLU(True),
            #nn.MaxPool2d(2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            #nn.BatchNorm2d(32),
            nn.ReLU(True),
            #nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(16),
            #nn.ReLU(True),
            #nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1, bias=False),
            #nn.BatchNorm2d(16),
            #nn.ReLU(True)
        )
        #self.fc1 = nn.Linear(8*8*16, 1024)
        #self.fc2 = nn.Linear(1024, 1024)

    def forward(self, x):
        out = self.conv_block(x).reshape(-1, 8*8*16)
        #out = self.fc1(out.reshape(-1, 8*8*16))
        return out


class Qhead(nn.Module):

    def __init__(self):
        super(Qhead, self).__init__()

        self.conv1 = nn.Conv2d(256, 128, 1, bias = False)
        self.bn1 = nn.BatchNorm2d(128)

        self.conv2 = nn.Conv2d(128, 10, 1)
    

    def forward(self, x):
        x = nn.functional.leaky_relu(self.bn1(self.conv1(x)), 0.2, inplace=True)
        logits = self.conv2(x).squeeze()
        return logits
    

class Feature_Decoder__(nn.Module):

    def __init__(self, in_channels, out_channels = 3, shape = 32):
        super(Feature_Decoder, self).__init__()
        if shape == 32:
            self.conv_block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride = 1, padding = 0),
                #nn.BatchNorm2d(out_channels)
            )
        elif shape == 16:
            self.conv_block = nn.Sequential(
                nn.Upsample(scale_factor = 2),
                nn.Conv2d(in_channels, out_channels, 3, stride = 1, padding = 1),
                #nn.BatchNorm2d(out_channels)
            )
        elif shape == 8:
            self.conv_block = nn.Sequential(
                nn.Upsample(scale_factor = 2),
                nn.Conv2d(in_channels, int(in_channels / 2), 3, stride = 1, padding = 1),
                nn.LeakyReLU(inplace = True),
                nn.BatchNorm2d(int(in_channels / 2)),
                nn.Upsample(scale_factor = 2),
                nn.Conv2d(int(in_channels / 2), out_channels, 3, stride = 1, padding = 1),
                #nn.BatchNorm2d(out_channels)
            )
        elif shape == 4:
            self.conv_block = nn.Sequential(
                nn.Upsample(scale_factor = 2),
                nn.Conv2d(in_channels, int(in_channels / 2), 3, stride = 1, padding = 1),
                nn.LeakyReLU(inplace = True),
                nn.BatchNorm2d(int(in_channels / 2)),
                nn.Upsample(scale_factor = 2),
                nn.Conv2d(int(in_channels / 2), int(in_channels / 4.0), 3, stride = 1, padding = 1),
                nn.LeakyReLU(inplace = True),
                nn.BatchNorm2d(int(in_channels / 4.0)),
                nn.Upsample(scale_factor = 2),
                nn.Conv2d(int(in_channels / 4.0), out_channels, 3, stride = 1, padding = 1),
                #nn.BatchNorm2d(out_channels)
            )
        for mod in self.conv_block:
            if isinstance(mod, nn.Conv2d):
                nn.init.kaiming_normal_(mod.weight.data)
                nn.init.constant_(mod.bias.data, 0.0)
            elif isinstance(mod, nn.BatchNorm2d):
                nn.init.normal_(mod.weight.data, 1.0, 0.02)
                nn.init.constant_(mod.bias.data, 0.0)
        
    def forward(self, x):
        out = self.conv_block(x)
        return out


class Feature_Decoder(nn.Module):

    def __init__(self, init = False, F_init = 'ortho'):
        super(Feature_Decoder, self).__init__()
        self.F_init = F_init
        self.upsample = nn.Upsample(scale_factor = 2)
        self.conv1 = nn.Conv2d(512, 256, 1, stride = 1, padding = 0)
        self.conv2 = nn.Conv2d(256, 128, 1, stride = 1, padding = 0)
        self.conv3 = nn.Conv2d(128, 64, 1, stride = 1, padding = 0)
        self.conv4 = nn.Conv2d(64, 3, 1, stride = 1, padding = 0)
        #self.conv5 = nn.Conv2d(64, 3, 1, stride = 1, padding = 0)
        self.conv6 = nn.Conv2d(3, 3, 1, stride = 1, padding = 0)
        #self.bn = nn.BatchNorm2d(3)
        #self.relu = nn.ReLU(inplace = True)
        #self.leaky = nn.LeakyReLU(0.2, inplace = True)
        self.conv_31 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv_32 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.conv_33 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv_34 = nn.Conv2d(3, 3, 3, stride=1, padding=1)

        #self.bn1 = nn.BatchNorm2d(256)
        self.bn1 = BatchInstanceNorm2d(256)
        #self.bn1 = layers.SandwichBatchNorm2d(256)
        #self.bn11 = nn.BatchNorm2d(256)
        self.bn11 = BatchInstanceNorm2d(256)
        #self.bn11 = layers.SandwichBatchNorm2d(256)
        #self.bn2 = nn.BatchNorm2d(128)
        self.bn2 = BatchInstanceNorm2d(128)
        #self.bn2 = layers.SandwichBatchNorm2d(128)
        #self.bn22 = nn.BatchNorm2d(128)
        self.bn22 = BatchInstanceNorm2d(128)
        #self.bn22 = layers.SandwichBatchNorm2d(128)
        #self.bn3 = nn.BatchNorm2d(64)
        self.bn3 = BatchInstanceNorm2d(64)
        #self.bn3 = layers.SandwichBatchNorm2d(64)
        #self.bn33 = nn.BatchNorm2d(64)
        self.bn33 = BatchInstanceNorm2d(64)
        #self.bn33 = layers.SandwichBatchNorm2d(64)
        #self.bn4 = nn.BatchNorm2d(3)
        self.bn4 = BatchInstanceNorm2d(3)
        #self.bn4 = layers.SandwichBatchNorm2d(3)
        #self.bn44 = nn.BatchNorm2d(3)
        self.bn44 = BatchInstanceNorm2d(3)
        #self.bn44 = layers.SandwichBatchNorm2d(3)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        
        if init:
            self.init_weights()

    def init_weights(self):
        self.nu = 0
        self.param_count = 0
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) or isinstance(module, nn.Embedding):
                if self.F_init == 'ortho':
                    nn.init.orthogonal_(module.weight)
                elif self.F_init == 'normal':
                    nn.init.normal_(module.weight, 0, 0.02)
                elif self.F_init == 'xavier':
                    nn.init.xavier_uniform_(module.weight)
                else:
                    raise ValueError('Invalid Initialization Method')
                self.param_count += sum([p.data.nelement() for p in module.parameters()])
            elif isinstance(module, nn.BatchNorm2d):
                self.nu += 1
        print('Param Count for FeatureDecoder initialized parameters : {:d}'.format(self.param_count))
        print('BatchNorm Layer : {:d}'.format(self.nu))


    def forward(self, x, f2, f3, f4, f5):
        out1 = (self.conv1(self.upsample(f5)))
        out2 = (self.conv_31(out1 + f4))
        out3 = (self.conv2(self.upsample(out2)))
        out4 = (self.conv_32(out3 + f3))
        out5 = (self.conv3(self.upsample(out4)))
      
        out6 = (self.conv_33(out5 + f2))
        out7 = (self.conv4(out6))
        #out2 = self.conv5(f1)
        #out_ = self.conv_34(out_ + out2)
        #out = self.bn((out + x))
        #out = self.relu(out)
        #out = self.leaky(out)
        out7 = nn.functional.sigmoid(out7)
        out8 = (x * out7)
        #out = self.conv_34(out)
        #print(out)
        #exit()
        #out = torch.nn.functional.tanh(out)
        #out = normalization(out)
        out = self.tanh(out8)
        return out, out7
