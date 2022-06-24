#!/bin/bash
#SBATCH --job-name=learnlet_exact_rec     # nom du job
#SBATCH --gpus 1
#SBATCH --partition shared-gpu
#SBATCH --mem 0
#SBATCH --ntasks=4                   # nombre de tâche MPI
#SBATCH --ntasks-per-node=4          # nombre de tâche MPI par noeud
#SBATCH --cpus-per-task=4           # nombre de coeurs à réserver par tâche
# /!\ Attention, la ligne suivante est trompeuse mais dans le vocabulaire
# de Slurm "multithread" fait bien référence à l'hyperthreading.
#SBATCH --hint=nomultithread         # on réserve des coeurs physiques et non logiques
#SBATCH --distribution=block:block   # on épingle les tâches sur des coeurs contigus
#SBATCH --time=12:00:00              # temps d’exécution maximum demande (HH:MM:SS)
#SBATCH --output=learnlet_exact_rec%j.out # nom du fichier de sortie
#SBATCH --error=learnlet_exact_rec%j.out  # nom du fichier d'erreur (ici commun avec la sortie)

set -x
cd $HOME/Thesis/understanding-unets

. ./submission_scripts/learnlets.sh

echo STARTING AT `date`
srun python ./learning_wavelets/training_scripts/learnlet_subclassed_training.py
echo FINISHED at `date`

wait  # wait for all ccc_mprun(s) to complete.
