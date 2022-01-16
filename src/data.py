import torch
import torchvision

import pandas as pd
import numpy as np

from PIL import Image

#basically this class instantiates x-ray/mask/transform data from a folder
#allows to extract it and perform transoformations/manipulations
class LungDataset(torch.utils.data.Dataset):
    def __init__(self, origin_mask_list, origins_folder, masks_folder, transforms=None):
        
        self.origin_mask_list = origin_mask_list
        self.origins_folder = origins_folder
        #a mask is binarized matrix where all pixels are set to 0 or 1
        self.masks_folder = masks_folder
        self.transforms = transforms
    
    #extracting a mask and its name from a file
    def __getitem__(self, idx):
        #extracting a mask and its name from a file
        origin_name, mask_name = self.origin_mask_list[idx]
        #loading image using the name
        origin = Image.open(self.origins_folder / (origin_name + ".png")).convert("P")
        #loading the mask
        mask = Image.open(self.masks_folder / (mask_name + ".png"))
        #if there was some transoform(i.e rotation, deformation, scaling, blur, etc)
        #we apply the reseptive transform
        if self.transforms is not None:
            origin, mask = self.transforms((origin, mask))

        #converting the the image of Image type to Tensor 
        #not sure why we subract 0.5 here ?
        origin = torchvision.transforms.functional.to_tensor(origin) - 0.5

        #converting Image mask to array
        mask = np.array(mask)
        #changing integer to 64 bits if longer than 128(what longer than 128?)
        mask = (torch.tensor(mask) > 128).long() 
        return origin, mask
        
    #return the number of images in the dataset
    def __len__(self):
        return len(self.origin_mask_list)

#performs padding on the image and the mask 
class Pad():
    #just reminder for myself - __init__ used to initialise newly created object
    def __init__(self, max_padding):
        self.max_padding = max_padding
    
    #__cal__implements function call operator
    #ex x = Pad()
    #orgin, maks = x(sample)
    def __call__(self, sample):
        origin, mask = sample
        padding = np.random.randint(0, self.max_padding)
#         origin = torchvision.transforms.functional.pad(origin, padding=padding, padding_mode="symmetric")
        origin = torchvision.transforms.functional.pad(origin, padding=padding, fill=0)
        mask = torchvision.transforms.functional.pad(mask, padding=padding, fill=0)
        return origin, mask

#used to crop image by picking random crop given max_shift
class Crop():
    def __init__(self, max_shift):
        self.max_shift = max_shift
    #sample is an array(or tensor) that contains mask and image
    def __call__(self, sample):
        origin, mask = sample
        tl_shift = np.random.randint(0, self.max_shift)
        br_shift = np.random.randint(0, self.max_shift)
        origin_w, origin_h = origin.size
        crop_w = origin_w - tl_shift - br_shift
        crop_h = origin_h - tl_shift - br_shift
        
        origin = torchvision.transforms.functional.crop(origin, tl_shift, tl_shift,
                                                        crop_h, crop_w)
        mask = torchvision.transforms.functional.crop(mask, tl_shift, tl_shift,
                                                        crop_h, crop_w)
        return origin, mask

#recisign the image and the mask to the desire output size
class Resize():
    def __init__(self, output_size):
        self.output_size = output_size
        
    def __call__(self, sample):
        origin, mask = sample
        origin = torchvision.transforms.functional.resize(origin, self.output_size)
        mask = torchvision.transforms.functional.resize(mask, self.output_size)
        
        return origin, mask

#blending 
#on high level this blends 1 or 2 masks with the actual image
#what's the purpose ? 
def blend(origin, mask1=None, mask2=None):
    img = torchvision.transforms.functional.to_pil_image(origin + 0.5).convert("RGB")
    if mask1 is not None:
        #torchvision.transforms.functional.to_pil_image - Convert a tensor or an ndarray to PIL Image
        #torch.cat Concatenates the given sequence of seq tensors in the given dimension
        #not exactly sure about the details
        #do we change 
        mask1 =  torchvision.transforms.functional.to_pil_image(torch.cat([
            #tensor.zeros_like - Returns a tensor filled with the scalar value 0, with the same size as input.
            torch.zeros_like(origin),
            #what's the purpose of the brackets - not sure
            torch.stack([mask1.float()]),
            torch.zeros_like(origin)
        ]))
        #here the blending
        img = Image.blend(img, mask1, 0.2)
        
    if mask2 is not None:
        mask2 =  torchvision.transforms.functional.to_pil_image(torch.cat([
            torch.stack([mask2.float()]),
            torch.zeros_like(origin),
            torch.zeros_like(origin)
        ]))
        img = Image.blend(img, mask2, 0.2)
    
    return img
