#!/bin/bash
#SBATCH -n _NPROC_
#SBATCH --time=_TIME_
#SBATCH --mem-per-cpu=_MEM_
#SBATCH -p batch
#SBATCH -o _OUT_FILE_
#SBATCH -e _ERR_FILE_

env > _ENV_FILE_

HOSTNAME=`hostname`
echo "Starting job at ${HOSTNAME}"
echo "* job id:     _JOBID_"
echo "* params:     _PARAMS_"
echo "* processes:  _NPROC_"
echo "* time limit: _TIME_"
echo "* memory lim: _MEM_"
echo "* work dir:   _DATA_DIR_"
echo "* slurm file: _SLURM_FILE_"
echo "* job file:   _JOB_FILE_"
echo "* out file:   _OUT_FILE_"
echo "* err file:   _ERR_FILE_"
echo "* env file:   _ENV_FILE_"
echo "---------------------------------------"
source _SCRIPT_DIR_/load_libs.sh
srun --mpi=pmi2 python3 _JOB_FILE_ _PARAMS_
source _SCRIPT_DIR_/unload_libs.sh
echo "---------------------------------------"
echo "Job ended"

