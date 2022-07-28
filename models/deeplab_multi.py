# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

affine_par = True

class GradientReverse(torch.autograd.Function):
    scale = 1.0

    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return GradientReverse.scale * grad_output.neg()
def grad_reverse(x, lambd=1.0):
    GradientReverse.scale = lambd
    return GradientReverse.apply(x)
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, bn_momentum=0.1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=1, stride=stride, bias=False)  # change
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)

        padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,  # change
                               padding=padding, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine=affine_par)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Classifier_Module(nn.Module):
    def __init__(self, inplanes, dilation_series, padding_series, num_classes):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                nn.Conv2d(inplanes, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True))
            #print(num_classes)
            #exit()
        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
            return out
class Classifier_Module2(nn.Module):
    def __init__(self, inplanes, dilation_series, padding_series, num_classes, droprate = 0.1, use_se = True):
        super(Classifier_Module2, self).__init__()
        self.conv2d_list = nn.ModuleList()
        self.conv2d_list.append(
                nn.Sequential(*[
                nn.Conv2d(inplanes, 256, kernel_size=1, stride=1, padding=0, dilation=1, bias=True),
                nn.GroupNorm(num_groups=32, num_channels=256, affine = True),
                nn.ReLU(inplace=True) ]))

        for dilation, padding in zip(dilation_series, padding_series):
            #self.conv2d_list.append(
            #    nn.BatchNorm2d(inplanes))
            self.conv2d_list.append(
                nn.Sequential(*[
                #nn.ReflectionPad2d(padding),
                nn.Conv2d(inplanes, 256, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True), 
                nn.GroupNorm(num_groups=32, num_channels=256, affine = True),
                nn.ReLU(inplace=True) ]))
 
        if use_se:
            self.bottleneck = nn.Sequential(*[SEBlock(256 * (len(dilation_series) + 1)),
                        nn.Conv2d(256 * (len(dilation_series) + 1), 256, kernel_size=3, stride=1, padding=1, dilation=1, bias=True) ,
                        nn.GroupNorm(num_groups=32, num_channels=256, affine = True) ])
        else:
            self.bottleneck = nn.Sequential(*[
                nn.Conv2d(256 * (len(dilation_series) + 1), 256, kernel_size=3, stride=1, padding=1, dilation=1, bias=True) ,
                nn.GroupNorm(num_groups=32, num_channels=256, affine = True) ])

        self.head = nn.Sequential(*[nn.Dropout2d(droprate),
            nn.Conv2d(256, num_classes, kernel_size=1, padding=0, dilation=1, bias=False) ])

        ##########init#######
        for m in self.conv2d_list:
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d) or isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        for m in self.bottleneck:
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d) or isinstance(m, nn.GroupNorm) or isinstance(m, nn.LayerNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        for m in self.head:
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.001)

    def forward(self, x, get_feat=False):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out = torch.cat( (out, self.conv2d_list[i+1](x)), 1)
        out = self.bottleneck(out)
        if get_feat:
            out_dict = {}
            out = self.head[0](out)
            out_dict['feat'] = out
            out = self.head[1](out)
            out_dict['out'] = out
            return out_dict
        else:
            out = self.head(out)
            return out
class SEBlock(nn.Module):
    def __init__(self, inplanes, r = 16):
        super(SEBlock, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        self.se = nn.Sequential(
                nn.Linear(inplanes, inplanes//r),
                nn.ReLU(inplace=True),
                nn.Linear(inplanes//r, inplanes),
                nn.Sigmoid()
        )
    def forward(self, x):
        xx = self.global_pool(x)
        xx = xx.view(xx.size(0), xx.size(1))
        se_weight = self.se(xx).unsqueeze(-1).unsqueeze(-1)
        return x.mul(se_weight)
class ResNetMulti(nn.Module):
    def __init__(self, block, layers, num_classes):
        self.inplanes = 64
        super(ResNetMulti, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, ceil_mode=True)  # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=1, dilation=4)
        self.layer5 = self._make_pred_layer(Classifier_Module, 1024, [6, 12, 18, 24], [
                                            6, 12, 18, 24], num_classes)
        self.layer6 = self._make_pred_layer(Classifier_Module, 2048, [6, 12, 18, 24], [
                                            6, 12, 18, 24], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # New! Rotation prediction head
        self.rotation_prediction_head = nn.Identity()
        self.proto_classifier=nn.Conv2d(2048, num_classes, kernel_size=1, stride=1, bias=False)
        self.fuse_weight_1 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.fuse_weight_1.data.fill_(2)
        self.a=[]
        
        self.para = nn.Parameter(torch.tensor([0.5]), requires_grad=True)
        self.para.data.fill_(2)
        self.a.append(self.para)
        self.a.append(self.fuse_weight_1)
    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, affine=affine_par))
        layers = []
        layers.append(block(self.inplanes, planes, stride,
                            dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def _make_pred_layer(self, block, inplanes, dilation_series, padding_series, num_classes):
        return block(inplanes, dilation_series, padding_series, num_classes)

    def forward(self, x):
        #print('we are ready')
        #exit()
        input_size = x.size()[2:]
        self.input_size=input_size
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # Resolution 1
        #x_norm=F.normalize(x)
        #exit()
        #x1 = self.layer5(20*x_norm)
        x1 = self.layer5(x)
        x1 = F.interpolate(x1, size=input_size,
                           mode='bilinear', align_corners=True)

        # Resolution 2
        x2 = self.layer4(x)
        #x2_f=F.normalize(x2)
        x2_f=x2
        #print(x2_f.shape,x2.shape)
        x2 = self.layer6(x2)
        #print(x2_f.shape,x2.shape)
        x2_nointer=x2
        #print(x2_nointer.shape)
        #exit()
        #print(x2_f.shape,x2.shape)
        #exit()
        x2 = F.interpolate(x2, size=input_size,
                           mode='bilinear', align_corners=True)
        #print(x2.shape,'lllllllllll')
        return x2, x1,x2_nointer,x2_f  # changed!
    def classifier(self, x, reverse=False, eta=0.1):
        if reverse:
            x = grad_reverse(x, eta)
        x = self.layer6(x)    
        return x # changed!
    def prototype(self, x):
        if True:
            x_logits = self.class_t(x) 
            logit_size=x_logits.size()
            prototype=self.proto_classifier.weight
            #print(prototype.shape)
            prototype=prototype.reshape(-1,prototype.shape[0],prototype.shape[1]).repeat(x.shape[0],1,1)#.detach()
            #prototype=F.normalize(prototype)#.detach()
            #print(prototype.shape)
            #print(x_logits.shape,logit_size)
            #print(x.shape)
            #exit()
            prob=F.softmax(x_logits.reshape(logit_size[0],logit_size[1],-1),dim=1).transpose(1,2).contiguous()
            #print(self.fuse_weight_1)
            #exit()
            #print(prob.shape,prototype.shape)
            #exit()
            x_s_t=torch.bmm(prob,self.para*prototype).transpose(1,2).contiguous()
            x_o = self.layer6(x_s_t.reshape(x.shape[0],x.shape[1],x.shape[2],x.shape[3]))
            #x_o = F.interpolate(x_o, size=self.input_size,
                               #mode='bilinear', align_corners=True)
            #print(x_s_t.shape,x.shape)
            #exit()
            #exit()
            #print(x_o.shape)
            #exit()
        else:
            w,h=x.shape[2],x.shape[3]
            prototype=self.proto_classifier.weight
            prototype_1=prototype.reshape(prototype.shape[0],prototype.shape[1],-1).transpose(0,2).contiguous().repeat(2,1,1)
            x=x.reshape(x.shape[0],x.shape[1],-1).transpose(1,2).contiguous()
            #print(x.shape,prototype.shape)
            #exit()
            x_o=torch.bmm(x,prototype_1).transpose(1,2).contiguous().reshape(x.shape[0],prototype_1.shape[2],w,h)
            x_o = F.interpolate(x_o, size=self.input_size,
                               mode='bilinear', align_corners=True)            
        return x_o # changed!    
    def class_t(self, x):
        #print(x.shape)
        #print(self.proto_classifier.weight)
        if True:
            w,h=x.shape[2],x.shape[3]
            prototype=self.proto_classifier.weight
            #prototype=F.normalize(prototype).reshape(prototype.shape[0],prototype.shape[1],-1).transpose(0,2).contiguous().repeat(x.shape[0],1,1)
            prototype=prototype.reshape(prototype.shape[0],prototype.shape[1],-1).transpose(0,2).contiguous().repeat(x.shape[0],1,1)
            x=x.reshape(x.shape[0],x.shape[1],-1).transpose(1,2).contiguous()
            #print(x.shape,prototype.shape,torch.bmm(x,prototype).shape)
            #exit()
            #print(x.shape,prototype.shape)
            
            x_logits=torch.bmm(x,prototype).transpose(1,2).contiguous().reshape(x.shape[0],prototype.shape[2],w,h)
            #print(x_logits.shape)
            #exit()
            #print(x.shape,prototype.shape)
            #exit()
            #print(prototype.shape,'class_t')
            #exit(0)
        else:
            x_logits = self.proto_classifier(x) 

        return x_logits # changed!        
    def get_1x_lr_params_NOscale(self):
        """
        This generator returns all the parameters of the net except for
        the last classification layer. Note that for each batchnorm layer,
        requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
        any batchnorm parameter
        """
        b = []

        b.append(self.conv1)
        b.append(self.bn1)
        b.append(self.layer1)
        b.append(self.layer2)
        b.append(self.layer3)
        b.append(self.layer4)

        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj += 1
                    if k.requires_grad:
                        yield k

    def get_10x_lr_params(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        """
        b = []
        b.append(self.layer5.parameters())
        b.append(self.layer6.parameters())
        b.append(self.proto_classifier.parameters())
        #b.append(self.para.parameters())
        b.append(self.a)
        for j in range(len(b)):
            for i in b[j]:
                yield i

    def optim_parameters(self, lr):
        return [{'params': self.get_1x_lr_params_NOscale(), 'lr': lr},
                {'params': self.get_10x_lr_params(), 'lr': 10 * lr}]


def DeeplabMulti(num_classes=21, init=None):

    # Create model
    model = ResNetMulti(Bottleneck, [3, 4, 23, 3], num_classes)

    # Standard DeepLabv2 initialization
    if init:
        saved_state_dict = torch.load(init)
        new_params = model.state_dict().copy()
        for i in saved_state_dict:
            i_parts = i.split('.')
            if not i_parts[1] == 'layer5':
                new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
        model.load_state_dict(new_params)

    return model
