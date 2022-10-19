#!/bin/bash
#SBATCH --job-name=learnlet_256     # nom du job
#SBATCH --gpus 1
#SBATCH --partition shared-gpu
#SBATCH --mem=12000
#SBATCH --time=0-12:00:00              # temps d’exécution maximum demande (HH:MM:SS)
#SBATCH --output=learnlet_256_150_%j.out # nom du fichier de sortie
#SBATCH --error=learnlet_256_150_%j.out  # nom du fichier d'erreur (ici commun avec la sortie)

set -x
cd $HOME/Thesis/understanding-unets

. ./submission_scripts/learnlets.sh

echo STARTING AT `date`
srun python ./learning_wavelets/training_scripts/learnlet_subclassed_training.py
echo FINISHED at `date`

wait  # wait for all ccc_mprun(s) to complete.
