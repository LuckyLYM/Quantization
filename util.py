import torch.nn as nn
import numpy


class BinOp():
    def __init__(self, model,binarize_first_layer=False, binarize_last_layer=False):
        # count the number of Conv2d
        count_Conv2d = 0
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                count_Conv2d = count_Conv2d + 1

        if binarize_first_layer==True:
            start_range=0
        else:
            start_range = 1

        if binarize_last_layer==True:
            end_range = count_Conv2d-1
        else:
            end_range = count_Conv2d-2

        # [1, count_Conv2d-2]
        self.bin_range = numpy.linspace(start_range,
                end_range, end_range-start_range+1)\
                        .astype('int').tolist()


        self.num_of_params = len(self.bin_range)
        self.saved_params = []      # unbinarized weights
        self.target_params = []     
        self.target_modules = []    # binarized weights
        index = -1

        # They don't binarize the first and the last convolutional layer
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                index = index + 1
                if index in self.bin_range:
                    tmp = m.weight.data.clone()
                    self.saved_params.append(tmp)
                    self.target_modules.append(m.weight)


    def binarization(self):
        self.meancenterConvParams()
        self.clampConvParams()
        self.save_params()
        self.binarizeConvParams()

    def meancenterConvParams(self):
        for index in range(self.num_of_params):
            s = self.target_modules[index].data.size()
            negMean = self.target_modules[index].data.mean(1, keepdim=True).\
                    mul(-1).expand_as(self.target_modules[index].data)
            self.target_modules[index].data = self.target_modules[index].data.add(negMean)

    # clamp conv weights into the range (-1,1)
    def clampConvParams(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data = \
                    self.target_modules[index].data.clamp(-1.0, 1.0)

    def save_params(self):
        for index in range(self.num_of_params):
            self.saved_params[index].copy_(self.target_modules[index].data)

    def binarizeConvParams(self):
        for index in range(self.num_of_params):
            # conv2d.weight (out_channel, in_channel, kernek_size[0], kernel_size[1])
            # nelement return the number of elements
            n = self.target_modules[index].data[0].nelement()
            # size of the tensor (,,,)
            s = self.target_modules[index].data.size()
            # 1 for L1-norm, 3 for what??
            # m is the scaling factor
            m = self.target_modules[index].data.norm(1, 3, keepdim=True)\
                    .sum(2, keepdim=True).sum(1, keepdim=True).div(n)
            self.target_modules[index].data = \
                    self.target_modules[index].data.sign().mul(m.expand(s))

    def restore(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(self.saved_params[index])

    # very interesting idea
    def updateBinaryGradWeight(self):
        for index in range(self.num_of_params):
            
            # the full-precision weights
            weight = self.target_modules[index].data
            n = weight[0].nelement()
            s = weight.size()
            m = weight.norm(1, 3, keepdim=True)\
                    .sum(2, keepdim=True).sum(1, keepdim=True).div(n).expand(s)
            # scaling factor

            m[weight.lt(-1.0)] = 0 
            m[weight.gt(1.0)] = 0
            # m = m.add(1.0/n).mul(1.0-1.0/s[1]).mul(n)
            # self.target_modules[index].grad.data = \
            #         self.target_modules[index].grad.data.mul(m)
            m = m.mul(self.target_modules[index].grad.data)

            m_add = weight.sign().mul(self.target_modules[index].grad.data)

            m_add = m_add.sum(3, keepdim=True)\
                    .sum(2, keepdim=True).sum(1, keepdim=True).div(n).expand(s)

            m_add = m_add.mul(weight.sign())

            self.target_modules[index].grad.data = m.add(m_add).mul(1.0-1.0/s[1]).mul(n)
