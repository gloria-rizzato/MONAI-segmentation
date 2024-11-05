# Inference on new 3DFLAIR data 

flag="yes"
while [ "$flag" == "yes" ]
do
	################################################################################
	#                                                                 SET SUBJECT ID
	################################################################################

	echo "Enter subject ID:" 
	read subj
	echo "============ Subject ID: $subj ============"

	################################################################################
	#                                                                  READING INPUT
	################################################################################

	ROOT_DIR="$(pwd)"
	FLAIR3D_FILE=${ROOT_DIR}/subjects/${subj}/raw_FLAIR.nii.gz

	echo "FLAIR file: ${FLAIR3D_FILE}"

	################################################################################
	#                                                          FLAIR3D PREPROCESSING
	################################################################################

	echo -e "\n"
	   echo "====================="
	   echo "FLAIR3D PREPROCESSING"
	   echo "Performing N4BC"

	N4SHRINK=4

	#Perform normalization and bias field correction

	N4BiasFieldCorrection   -d 3 \
		                -i ${FLAIR3D_FILE} \
		                -s ${N4SHRINK} \
		                -o ${ROOT_DIR}/subjects/${subj}/raw_FLAIR3D_N4BC_${subj}.nii.gz
	echo -e "\n"
	echo "N4BC done"
	################################################################################
	#                                                          		     BET
	################################################################################


	echo -e "\n"
	   echo "====================="
	   echo "BET"
	bet ${ROOT_DIR}/subjects/${subj}/raw_FLAIR3D_N4BC_${subj}.nii.gz ${ROOT_DIR}/subjects/${subj}/raw_FLAIR3D_N4BC_bet.nii.gz -A
	fslmaths ${ROOT_DIR}/subjects/${subj}/raw_FLAIR3D_N4BC_bet_outskull_mesh.nii.gz -s 0.4 ${ROOT_DIR}/subjects/${subj}/raw_FLAIR3D_N4BC_mesh.nii.gz
	fslmaths ${ROOT_DIR}/subjects/${subj}/raw_FLAIR3D_N4BC_mesh.nii.gz -bin ${ROOT_DIR}/subjects/${subj}/raw_FLAIR3D_N4BC_mesh.nii.gz

	echo -e "\n"
	echo "BET done"
	################################################################################
	#                                                              CROPPING FLAIR 60
	################################################################################

	echo -e "\n"
	   echo "====================="
	   echo "CROPPING FLAIR for inference"
	python ${ROOT_DIR}/scripts/processing_FLAIR3D.py "$ROOT_DIR" "$subj"

	################################################################################
	#                                                                  NET INFERENCE
	################################################################################

	echo -e "\n"
	   echo "====================="
	   echo "NET inference"
	python ${ROOT_DIR}/scripts/inference_newdata.py "$ROOT_DIR" "$subj"


	################################################################################
	#                                                                     PSD VOLUME
	################################################################################

	echo -e "\n"
	   echo "====================="
	   echo "PSD volume calculation"

	#Create a .txt file to save PSD volumes for each test subject

	echo "PSD volume [mm^3]" > ${ROOT_DIR}/PSDmasksonFLAIR/${subj}/PSDvol.txt

	#PSD volume calculation
	fslstats ${ROOT_DIR}/PSDmasksonFLAIR/${subj}/PSDonFLAIR_${subj}.nii.gz -V >> ${ROOT_DIR}/PSDmasksonFLAIR/${subj}/PSDvol.txt
	echo "PSD volume for subject ${subj}: "
	cat ${ROOT_DIR}/PSDmasksonFLAIR/${subj}/PSDvol.txt

	echo -e "\n"
	echo "============ Done for Subject ID: $subj ============"

        echo -e "\n\n\n"
	echo "====================="
	echo "Do you want to segment another subject? (yes/no)"
	read flag
done
echo -e "\n\n\n"
	echo "====================="
	echo "PROCESS COMPLETED"
	echo "====================="
    



