#!/bin/bash

FILEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && readlink -f -- . )"
source "${FILEDIR}/parameters.sh"

# default values
TIME="1-00:00:00"
MEM="1000"
DATETIME=`date +%F_%T`
JOBID="run_at_${DATETIME}"
JOBFILE="job.py"
NPROC=1
PARAMS=""

# process command line parameters
# http://stackoverflow.com/questions/192249/how-do-i-parse-command-line-arguments-in-bash
while [[ $# > 1 ]]
do
key="$1"

case $key in
    -t|--time)
    TIME="$2"
    shift
    ;;
    -m|--mem)
    MEM="$2"
    shift
    ;;
    -i|--jobid)
    JOBID="$2"
    shift
    ;;
    -n|--nproc)
    NPROC="$2"
    shift
    ;;
    -j|--jobfile)
    JOBFILE="$2"
    shift
    ;;
    -p|--params)
    shift
    PARAMS="$@"
    break
    ;;
    *)
    # unknown option
    echo "usage: ./run_experiment_slurm.sh -t <max_time> -m <max_mem> -i <job_identifier> -n <number_of_processes> -j <job_file> -p <parameters to be give to job file>"
    exit
    ;;
esac
shift
done

IDENTIFIER="${JOBID}"
EXEC_DIR=`readlink -f -- .`
DATA_DIR="${ELFIE_RESULTS}/${IDENTIFIER}"

if [ ! -f $JOBFILE ];
then
    echo "Could not locate '${JOBFILE}'"
    exit 1
fi

if [ -d $DATA_DIR ];
then
    echo "Folder already exists! Press 'y' to continue."
    read -n 1 cont
    if [ "$cont" != "y" ]; then
        echo " terminating"
        exit 0
    fi
    echo " continuing experiment"
fi
mkdir -p $DATA_DIR

echo "Running experiment with"
echo "* JOBID:     ${JOBID}"
echo "* PARAMS:    ${PARAMS}"
echo "* TIME:      ${TIME}"
echo "* MEMORY:    ${MEM}"
echo "* NPROC:     ${NPROC}"
echo "* DIR:       ${DATA_DIR}"

# create job script file and param file
SLURM_FILE="${DATA_DIR}/${JOBID}.sh"
JOB_FILE="${DATA_DIR}/job.py"
OUT_FILE="${DATA_DIR}/out.txt"
ERR_FILE="${DATA_DIR}/err.txt"
ENV_FILE="${DATA_DIR}/env.txt"
cp ${JOBFILE} ${JOB_FILE}
cat "${FILEDIR}/slurm_job_template.sh" |
    sed "s;_TIME_;${TIME};g" |
    sed "s;_MEM_;${MEM};g" |
    sed "s;_DATA_DIR_;${DATA_DIR};g" |
    sed "s;_SLURM_FILE_;${SLURM_FILE};g" |
    sed "s;_JOB_FILE_;${JOB_FILE};g" |
    sed "s;_OUT_FILE_;${OUT_FILE};g" |
    sed "s;_ERR_FILE_;${ERR_FILE};g" |
    sed "s;_ENV_FILE_;${ENV_FILE};g" |
    sed "s;_JOBID_;${JOBID};g" |
    sed "s;_PARAMS_;${PARAMS};g" |
    sed "s;_SCRIPT_DIR_;${FILEDIR};g" |
    sed "s;_NPROC_;${NPROC};g" > $SLURM_FILE
chmod ugo+x $SLURM_FILE

GIT_FILE="${DATA_DIR}/git.txt"
for r in "${ELFIE_GIT_REPOS[@]}"
do
    echo "Entering repository ${r}" >> ${GIT_FILE}
    cd ${r}
    echo "Current commit:" >> ${GIT_FILE}
    git log -n 1 >> ${GIT_FILE}
    echo "----" >> ${GIT_FILE}
    echo "Git status:" >> ${GIT_FILE}
    git status >> ${GIT_FILE}
    echo "----" >> ${GIT_FILE}
    echo "Local changes:" >> ${GIT_FILE}
    git diff >> ${GIT_FILE}
    echo "----" >> ${GIT_FILE}
    echo "Cached local changes:" >> ${GIT_FILE}
    git diff --cached >> ${GIT_FILE}
    echo "----" >> ${GIT_FILE}
    echo "end." >> ${GIT_FILE}
done

# add job to slurm queue
cd $DATA_DIR
sbatch $SLURM_FILE

