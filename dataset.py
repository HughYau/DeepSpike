import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pdb

class spikeData(Dataset):
    def __init__(self,data,event_index=None,target=None,normalize=False, shuffle=False,
                 transform=False,mask=False,flip=False,jigsaw=False,evMask=None,thresh=0):
        super().__init__()

        self.transform = transform
        self.flip = flip
        self.slice = 8
        self.jigsaw = jigsaw
        self.mask = mask
        self.shuffle = shuffle

        if self.flip:
            print("Using flip pretext")
        elif self.shuffle:
            print("Using shuffle pretext")
        elif self.jigsaw:
            print("Using jigsaw pretext")
        elif self.transform:
            print("Using random transform pretext")
        elif self.mask:
            print("Using mask pretext")
        else: 
            print("No pretext tasks!")

        ## Load from numpy
        data = torch.FloatTensor(data) ############
        data = data.view(data.shape[0],-1)  ###################

        ### Drop clustered events
        data = data[evMask == 0]
        event_index = event_index[evMask == 0]

        ### Choose events above thresh

        self.data = data[data.max(1)[0] >= thresh]
        self.event_index = event_index[data.max(1)[0] >= thresh]
        print("Found %d events at %.2f threshold"%(len(self.data),thresh))

        ### Normalize intensities to be between 0-1
        if normalize:
            self.data = self.data/self.data.max()

        self.event_index = torch.LongTensor(self.event_index)
        ## Collapse the H,W dimensions to a single F dimemsion
        self.H = self.data.shape[-1]
        self.size = self.data.shape[-1]
        ### Add a channel dimension
        self.data = self.data.unsqueeze(1)
        self.mask_tensor = torch.ones((1,self.size))
        self.mask_tensor[0,self.size//4: (self.size//4 + self.size//4)] = 0
        # self.mask_tensor = torch.cat((torch.ones((1,self.size//3)),
        #                               torch.zeros((1,self.size//3)),torch.ones((1,self.size//3))),dim=1)

        ## Make a tensor target
        if target is not None:
            self.target = torch.FloatTensor(target)

    def __len__(self):
        ### Method to return number of data points
        return len(self.data)

    def __getitem__(self,index):
        ### Method to fetch indexed element
#         pdb.set_trace()
        if self.transform and torch.rand(1) > 0.5:
            idx = int(np.random.randint(0,self.size,1))
            return torch.roll(self.data[index],idx), self.data[index], self.event_index[index]
        elif self.mask:
            return self.data[index]*(self.mask_tensor), self.data[index], self.event_index[index]
        elif self.flip:
            return torch.fliplr(self.data[index]), self.data[index], self.event_index[index]
        elif self.shuffle:
            idx = np.random.permutation(self.H)
            return self.data[index][0,idx].unsqueeze(0), self.data[index], self.event_index[index]
        elif self.jigsaw:
            idx = np.random.permutation(self.slice)
            return self.data[index].view(-1,self.size//self.slice)[idx].view(-1,self.size), self.data[index], self.event_index[index]
        else:
            return self.data[index], self.data[index], self.event_index[index]



