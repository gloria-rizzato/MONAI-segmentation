#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 15:52:21 2024

@author: gloria.rizzato

Apply 3D UNet to segment PSD from 3DFLAIR already prepaired
input: 3D FLAIR 60° cropped
output: PSD mask 60° cropped + PSD mask total  
"""

# Import libraries

from monai.utils import first, set_determinism
from monai.transforms import (
    AsDiscreted,
    Compose,
    EnsureChannelFirstd,
    Invertd,
    LoadImage,
    LoadImaged,
    SaveImage,
    ScaleIntensityd,
    Spacingd,
)
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.handlers.utils import from_engine
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.config import print_config, print_gpu_info
import torch
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import os
import glob
import sys
import pandas as pd

#print_config()

# Set up GPU usage 

#print("\n#### GPU INFORMATION ###")
#print_gpu_info()
device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
#print('Device:', device)
#Additional Info when using cuda
if device.type == 'cuda':
    #print(torch.cuda.get_device_name(0))
    torch.cuda.empty_cache()

# Define functions
def load_image(path, image):
    image_file = os.path.join(path, image)
    nib_image = nib.load(image_file)
    header_image = nib_image.header
    image = nib_image.get_fdata() 
    return image, header_image, nib_image

# Define transforms 

org_transforms = Compose(
    [
        LoadImaged(keys="image"),
        EnsureChannelFirstd(keys="image"),
        Spacingd(keys=["image"], pixdim=(0.5, 0.5, 0.5), mode="bilinear"),
        ScaleIntensityd(keys="image", minv=0, maxv=1.0),
    ]
)

post_transforms = Compose(
    [
        Invertd(
            keys="pred",
            transform=org_transforms,
            orig_keys="image",
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=False,
            to_tensor=True,
        ),
        AsDiscreted(keys="pred", argmax=True, to_onehot=2),
    ]
)

save_image=SaveImage(output_dir=os.path.join(str(sys.argv[1]), "PSDmasks", str(sys.argv[2])),  output_postfix="mask", resample=False, separate_folder=False, print_log=False) 


# Define model

model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128, 256),
    strides=(2,2,2,2),
    num_res_units=2,
    norm=Norm.BATCH,
).to(device)
#print(model)

# Define paths and loading data
def main():
    print("PSD mask creation and PSD volume calculation")
    FLAIRdir = os.path.join(str(sys.argv[1]), 'FLAIRnet') 
    model_dir = os.path.join(str(sys.argv[1]), 'scripts')
    FLAIRfile = sorted(glob.glob(os.path.join(FLAIRdir, "raw_FLAIR3D_N4BC_60cropped_" + str(sys.argv[2]) + ".nii.gz")))
    FLAIRdict = [{"image": image_name} for image_name in FLAIRfile]
    print("*** Loading data...")
    org_ds = CacheDataset(data=FLAIRdict, transform=org_transforms, cache_rate=1.0, num_workers=4, progress=False)
    org_loader = DataLoader(org_ds, batch_size=1, num_workers=4)
    
    #Apply Net to segment PSD 
    
    loader = LoadImage()
    
    model.load_state_dict(torch.load(os.path.join(model_dir, 'model.pth')))
    model.eval()
    print("*** Saving data...")
    with torch.no_grad():
        for data_new in org_loader:
            data_inputs = data_new["image"].to(device)
            roi_size = (160, 160, 160)
            sw_batch_size = 4
            data_new["pred"] = sliding_window_inference(data_inputs, roi_size, sw_batch_size, model)
            data_new = [post_transforms(i) for i in decollate_batch(data_new)]
            data_output = from_engine(["pred"])(data_new)
            data_label = data_output[0][1,:,:,:]
            save_image(data_label)
            # uncomment the following lines to visualize the predicted results
            # original_image = loader(test_output[0].meta["filename_or_obj"])
            # plt.figure("check", (8, 6))
            # plt.subplot(1, 2, 1)
            # plt.title("FLAIR")
            # plt.imshow(original_image[:, 30, :], cmap="gray")
            # plt.subplot(1, 2, 2)
            # plt.title("mask pred")
            # plt.imshow(test_output[0].detach().cpu()[1, :, 30, :])
            # plt.show()
        
    # Reconstruct PSD mask on FLAIR
    
    FLAIRtotdir = os.path.join(str(sys.argv[1]), 'subjects', str(sys.argv[2])) 
    PSDmaskdir = os.path.join(str(sys.argv[1]), 'PSDmasks', str(sys.argv[2])) 
    MASKtotdir = os.path.join(str(sys.argv[1]), 'PSDmasksonFLAIR')
    ACdir = os.path.join(str(sys.argv[1]), 'subjects', str(sys.argv[2])) 
    CSVdir = os.path.join(str(sys.argv[1]), 'subjects', str(sys.argv[2]))
    
    FLAIRtot, header_FLAIR, nib_FLAIR = load_image(FLAIRtotdir, 'raw_FLAIR3D_N4BC_' + str(sys.argv[2]) + '.nii.gz')
    PSDmask, header_PSDmask, nib_PSDmask = load_image(PSDmaskdir, 'raw_FLAIR3D_N4BC_60cropped_' + str(sys.argv[2]) + '_mask.nii.gz') 
    AC_mask, header_AC_mask, nib_AC_mask = load_image(ACdir, 'AC.nii.gz')
    AC_coordinates = np.asarray(np.where(AC_mask != 0))
    y_AC = AC_coordinates[2][0]
    csv_file = pd.read_csv(os.path.join(CSVdir, 'csvSlices.csv'), sep=',', header=0, nrows=1)
    df = pd.DataFrame(csv_file, columns=["anteriorSlice", "posteriorSlice"])
    mask3D = np.zeros((FLAIRtot.shape[0], FLAIRtot.shape[1], FLAIRtot.shape[2]))
    index = 1
    k=0
    
    for n_slice in range(df['posteriorSlice'][0], df['anteriorSlice'][0]):
        mask3D[:,n_slice, y_AC:FLAIRtot.shape[2]] = PSDmask[:,k,:]
        k = k+1
    mask_to_save = nib.Nifti1Image(mask3D, affine=nib_FLAIR.affine)
    print('Saving PSDmask on FLAIR')
    MASKtot_name = 'PSDonFLAIR_' + str(sys.argv[2]) + '.nii.gz'
    final_path = MASKtotdir + "/" + str(sys.argv[2])
    if not os.path.exists(final_path):
      os.makedirs(final_path)
    nib.save(mask_to_save, os.path.join(final_path,  MASKtot_name))
    print(f"Process completed for subject {str(sys.argv[2])}")

if __name__ == "__main__":
    main()

