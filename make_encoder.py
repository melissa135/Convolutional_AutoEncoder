import os.path
import torch
import collections
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from define_network import Compression_encoder,AutoEncoder
from sample_set import Sample_set

if __name__ == '__main__':

    path_ = os.path.abspath('.')

    fname = path_ + '/conv_autoencoder.pth'
    ae = AutoEncoder()
    ae.load_state_dict(torch.load(fname))
    #print ae.state_dict()
    ce = Compression_encoder()
    new_dict = collections.OrderedDict()

    new_dict['encoder.0.weight'] = ae.state_dict()['encoder.0.weight']
    new_dict['encoder.0.bias'] = ae.state_dict()['encoder.0.bias']
    new_dict['encoder.2.weight'] = ae.state_dict()['encoder.2.weight']
    new_dict['encoder.2.bias'] = ae.state_dict()['encoder.2.bias']
    new_dict['encoder.4.weight'] = ae.state_dict()['encoder.4.weight']
    new_dict['encoder.4.bias'] = ae.state_dict()['encoder.4.bias']

    ce.load_state_dict(new_dict)

    torch.save(ce.state_dict(),path_+'/compression_encoder.pth')
