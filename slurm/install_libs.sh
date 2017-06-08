#!/bin/bash

FILEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && readlink -f -- . )"
source "${FILEDIR}/parameters.sh"
cd $FILEDIR/..

if [[ "$1" == "--reinstall" ]];
then
    echo "Removing current installation.."
    rm -rf $ELFIE_VENV
fi

source ${FILEDIR}/load_libs.sh --only-modules

if [ ! -d $ELFIE_VENV ];
then
    command -v virtualenv >/dev/null 2>&1 ||
        { echo "Virtualenv is required but not installed. Try: sudo pip install virtualenv"; exit 1; }
    echo "No virtualenv found, creating.."
    virtualenv -p python3 $ELFIE_VENV --system-site-packages
fi

source ${FILEDIR}/load_libs.sh --only-venv

echo "Checking pip.."
piploc="`which pip`"
target="${ELFIE_VENV}/bin/pip"
if [[ $piploc != *${target} ]];
then
    echo "Error: Pip is currently in ${piploc} instead of *${target}?"
    echo "Deactivating virtualenv.."
    deactivate
    echo "Aborted."
    exit 1
fi

echo "Removing old builds.."
rm -rf "${ELFIE_VENV}/build"

echo "Setting environment varibles.."
unset CFLAGS
unset CXXFLAGS
unset LDFLAGS

echo "Installing ELFI.."
if [ -z "${ELFIE_SOURCE}" ];
then
    cd ${ELFIE_SOURCE}
    pip install numpy cython
    pip install -e .
else
    pip install elfi
fi

echo "Finding additional requirements.."
for r in "${ELFIE_REQUIREMENTS[@]}"
do
    n_requirements="$(cat ${r} | wc -l)"
    if [ $n_requirements -gt "0" ];
    then
        echo "Installing requirements from ${r}.."
        pip install -r $r
    else
        echo "Requirements ${r} is empty"
    fi
done

source ${FILEDIR}/unload_libs.sh

echo "Done."
