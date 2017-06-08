#!/bin/bash

FILEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "${FILEDIR}/parameters.sh"

if [[ "$1" != "--only-venv" ]];
then
    >&2 echo "Loading modules.."
    for m in "${ELFIE_MODULES[@]}"
    do
        module load ${m}
    done
    module list
fi

if [[ "$1" != "--only-modules" ]];
then
    >&2 echo "Activating virtualenv.."
    source "${ELFIE_VENV}/bin/activate"
fi

