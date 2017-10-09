import os
import torch
import random
import torch.utils.data as data
from pandas.io.parsers import read_csv


def read_files(folder):
    
    daily_data = []

    for root, _, fnames in os.walk(folder):
        for fname in fnames:
            
            path = os.path.join(root, fname)
            df = read_csv(path)
            
            for i in range(0,len(df)):
		if i % 48 == 0 :
		    if i != 0 :
		        daily_data.append(temp[:])
		    temp = [ df['price_change'][i] ]
		else :
		    temp.append(df['price_change'][i])

    return daily_data


class Sample_set(data.Dataset):

    def __init__(self, folder):

	data = read_files(folder)
	print 'This set contains %d items.' % len(data)
	self.data = data

    def __getitem__(self, index):

        item = torch.Tensor([self.data[index][:]]) # add one more dimension
        target = torch.Tensor([self.data[index][:]]) # add one more dimension

	return item,target

    def __len__(self):
        return len(self.data)
