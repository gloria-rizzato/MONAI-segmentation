# Parasagittal dura (PSD) segmentation tool 

The project aimed to segment parasagittal dura (PSD) from 3D-FLAIR sequence in a pediatric population (2-10 years old).
The algorithm was developed in [**MONAI**](https://monai.io/) which is a freely available, community-supported, PyTorch-based framework for deep learning in healthcare imaging.

A pre-trained 3D-UNet is provided and it can be used for inference on new data using the scripts model.pth in [scripts](https://github.com/gloria-rizzato/MONAI-segmentation/tree/main/src/scripts). 
Please, make sure you have downloaded the folder [scripts](https://github.com/gloria-rizzato/MONAI-segmentation/tree/main/src/scripts) for inference on new data.

<p align="center">
  <img src="https://github.com/gloria-rizzato/MONAI-segmentation/blob/main/PSD.gif"/>
</p>

## Data

To segment PSD and to calculate its volume, please make a main directory (called for example *PSD_inference*). In this main directory insert the script [inference_final.sh](https://github.com/gloria-rizzato/MONAI-segmentation/blob/main/inference_final.sh) and create 2 sub-directories called *scripts* and *subjects*, respectively. In the *scripts* directory make sure to insert all the required scripts (see folder [scripts](https://github.com/gloria-rizzato/MONAI-segmentation/tree/main/src/scripts)). In the *subjects*  create new sub-directories (for example, you can call the sub-directories with the ID of the subjects) and insert the following files:

- Raw 3D-FLAIR sequence (named *raw_FLAIR*) in .nii.gz format
- AC, PC masks (named *AC* and *PC*, respectively) in .nii.gz format

Your folders will be organized in this way, for example:
![folder organization](https://github.com/user-attachments/assets/594b2b22-0592-45bf-b323-51c194c5d1d4)

```
PSD_inference
│
|   inference_final.sh
|
└───scripts
|   |    model.pth
|   |    inference_newdata.py
|   |    processing_FLAIR3D.sh
|
└───subjects
    └───SUB01
    |   |    raw_FLAIR.nii.gz
    |   |    AC.nii.gz
    |   |    PC.nii.gz
    └───SUB02
    |   |    raw_FLAIR.nii.gz
    |   |    AC.nii.gz
    |   |    PC.nii.gz
    |   ...

```

## Requirements

python 3 is required and `python 3.10.13` was used in the project.

Requirements can be found at requirements.txt (mettere link e txt file)

Please use ```pip install requirements.txt``` to install the requirements

## How to run the code

Now you are ready to run the code.
Open a new terminal, activate the environment, set the main directory (for example PSD_inference) and launch the script:
```
sh inference_final.sh
```
Type in the subject ID and the inference will start. When the process is completed you can either choose to make inference on a new subject or quit. 

## Output 

After the inference on new data, 3 sub-directories will be created in the main directory:
- FLAIRnet, storing the 3DFLAIR taken as input by the algorithm to segment PSD
- PSDmasks, storing the PSD segmentation
- PSDmasksonFLAIR, storing the PSD segmentation which can be overlapped to the original 3DFLAIR and a .txt file containing the information related to the volume of the PSD

At the end you will have, for example:
![outputfolders](https://github.com/user-attachments/assets/bad6dd89-48fd-4294-baf7-7aec02a19ec8)
