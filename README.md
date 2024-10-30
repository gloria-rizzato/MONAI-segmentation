# Parasagittal dura (PSD) segmentation tool 

The project aimed to segment parasagittal dura (PSD) from 3D-FLAIR sequence in a pediatric population (2-10 years old).
The algorithm was developed in [**MONAI**<br>](https://monai.io/) which is a freely available, community-supported, PyTorch-based framework for deep learning in healthcare imaging.

A pre-trained 3D-UNet is provided and it can be used for inference on new data using the scripts in (mettere il path alla cartella che ci sarà su github). 
Please, make sure you have downloaded the folder (link alla folder) for inference on new data.



## Data

To segment PSD and to calculate its volume, please make a main directory (called for example *PSD_inference*). In this main directory create 2 sub-directories called *subjects* and *scripts*, respectively. In the *scripts* directory make sure to insert all the required scripts (see folder e mettere link alla folder su github). In the *subjects*  create new sub-directories (for example, you can call the sub-directories with the ID of the subjects) and insert the following files:

- Raw 3D-FLAIR sequence (named *raw_FLAIR*) in .nii.gz format
- AC, PC masks (named *AC* and *PC*, respectively) in .nii.gz format
