#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 11:08:53 2024

@author: gloria.rizzato

Find anterior and posterior slices to cut FLAIR3D after N4BC + cropping images
input: subject ID
output: cropped FLAIRs (60° + AC in coronal plane)
"""

# Import libraries
import os
import sys
import nibabel as nib
import numpy as np
import matplotlib as plt
import pandas as pd


# Define function
def load_image(path, image):
    image_file = os.path.join(path, image)
    nib_image = nib.load(image_file)
    header_image = nib_image.header
    image = nib_image.get_fdata() 
    return image, header_image, nib_image

def main():
    print("Finding anterior and posterior slices + cropping FLAIR3D")
    root_dir=os.path.join(str(sys.argv[1]), 'subjects', str(sys.argv[2]))
    root_output = os.path.join(str(sys.argv[1]), 'FLAIRnet')
    if not os.path.exists(root_output):
      os.makedirs(root_output)
      print("Directory created successfully!")
    FLAIR, headerFLAIR, nibFLAIR = load_image(root_dir, "raw_FLAIR3D_N4BC_" + str(sys.argv[2]) + ".nii.gz")
    #AC-PC points manually defined
    AC, headerAC, nibAC = load_image(root_dir, "AC.nii.gz")
    PC, headerPC, nibPC = load_image(root_dir, "PC.nii.gz")
    #fslbet on 3D FLAIR image
    BET, headerBET, nibBET = load_image(root_dir, "raw_FLAIR3D_N4BC_bet.nii.gz")
    #Dorsal profile (smoothed outskull mesh from fslbet)
    MESH, headerMESH, nibMESH = load_image(root_dir, "raw_FLAIR3D_N4BC_mesh.nii.gz")
    
    # AC and PC coordinate definition:
    idAC = np.nonzero(AC)
    zAC, yAC, xAC = idAC[0][0], idAC[1][0], idAC[2][0]
    idPC = np.nonzero(PC)
    zPC, yPC, xPC = idPC[0][0], idPC[1][0], idPC[2][0]

    
    # Geometric constraint imposition to define the PSD dorsal profile:
    xM = (xAC + xPC) / 2
    yM = (yAC + yPC) / 2
    d = (3/4) * ((xPC - xAC)**2 + (yPC - yAC)**2)
    coeff = -(xPC - xAC) / (yPC - yAC)
    a = 1 + (coeff)**2
    b = -2 * xM * a
    c = (xM)**2 + (coeff * xM)**2 - d
    roots = np.roots([a, b, c])
    xC = round(min(roots))
    yC = round(coeff * (xC - xM) + yM)
    
    # Line passing through C (defined above) and AC-PC definition:
    zero = np.zeros((FLAIR.shape[1], FLAIR.shape[2]))
    for x in range(FLAIR.shape[2]):
        y1 = round((yC - yAC) / (xC - xAC) * (x - xAC) + yAC)
        if 0 < y1 < FLAIR.shape[2]:
            zero[y1, x] = 1
        y2 = round((yC - yPC) / (xC - xPC) * (x - xPC) + yPC)
        if 0 < y2 < FLAIR.shape[2]:
            zero[y2, x] = 1
            
    # Dorsal starting (post) and ending (ant) point:
    MESH_slice = np.squeeze(MESH[round((MESH.shape[1])/2), :, :])
    TOT = zero + MESH_slice
    # plt.imshow(TOT, aspect='auto')
    # plt.colorbar()
    # plt.show()
        
    row, col = np.where(TOT > 1)
    post = np.min(row) - 1
    ant = np.max(row) - 1
    print(f"\nAnterior slice:  {ant} \nPosterior slice: {post}\n")
    df_slices = pd.DataFrame(list(zip([ant], [post])), columns=["anteriorSlice", "posteriorSlice"])
    df_slices.to_csv(os.path.join(root_dir, 'csvSlices.csv'), index=False, sep = ',')
    FLAIR60 = np.zeros((FLAIR.shape[0], ant-post, FLAIR.shape[2]))
    FLAIR60 = FLAIR[:, post:ant, :]
    print("Saving 60° FLAIR")
    FLAIR60nifti = nib.Nifti1Image(FLAIR60, affine=nibFLAIR.affine)
    nib.save(FLAIR60nifti, os.path.join(root_dir, "raw_FLAIR3D_N4BC_60_" + str(sys.argv[2]) + ".nii.gz"))
    FLAIRcropped = np.zeros((np.array(FLAIR60).shape[0], np.array(FLAIR60).shape[1], np.array(FLAIR60).shape[2] - xAC))
    for sl in range(0, FLAIR60.shape[1]):
        FLAIRcropped[:, sl ,:] = FLAIR60[:, sl, xAC: FLAIR60.shape[2]]
    print("Saving 60° cropped FLAIR")
    FLAIRcroppednifti = nib.Nifti1Image(FLAIRcropped, affine=nibFLAIR.affine)
    nib.save(FLAIRcroppednifti, os.path.join(root_dir, "raw_FLAIR3D_N4BC_60cropped_" + str(sys.argv[2]) + ".nii.gz"))
    nib.save(FLAIRcroppednifti, os.path.join(root_output, "raw_FLAIR3D_N4BC_60cropped_" + str(sys.argv[2]) + ".nii.gz"))
if __name__ == "__main__":
    main()
