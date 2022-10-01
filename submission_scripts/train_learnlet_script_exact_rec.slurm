#!/bin/bash
#SBATCH --job-name=learnlet_256_500     # nom du job
#SBATCH --gpus 1
#SBATCH --partition debug-gpu
#SBATCH --mem 0
#SBATCH --time=0-00:15:00              # temps d’exécution maximum demande (HH:MM:SS) 35
#SBATCH --output=learnlet_256_500_%j.out # nom du fichier de sortie
#SBATCH --error=learnlet_256_500_%j.out  # nom du fichier d'erreur (ici commun avec la sortie)

set -x
cd $HOME/Thesis/understanding-unets

. ./submission_scripts/learnlets.sh

echo STARTING AT `date`
srun python ./learning_wavelets/training_scripts/learnlet_subclassed_training.py
echo FINISHED at `date`

wait  # wait for all ccc_mprun(s) to complete.
