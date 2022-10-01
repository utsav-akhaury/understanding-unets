#!/bin/bash
#SBATCH --job-name=unet     # nom du job
#SBATCH --gpus 1
#SBATCH --partition shared-gpu
#SBATCH --mem 0
#SBATCH --time=0-08:00:00              # temps d’exécution maximum demande (HH:MM:SS)
#SBATCH --output=unet_%j.out # nom du fichier de sortie
#SBATCH --error=unet_%j.out  # nom du fichier d'erreur (ici commun avec la sortie)

set -x
cd $HOME/Thesis/understanding-unets

. ./submission_scripts/learnlets.sh

echo STARTING AT `date`
srun python ./learning_wavelets/training_scripts/bias_free_unet_training.py
echo FINISHED at `date`

wait  # wait for all ccc_mprun(s) to complete.
