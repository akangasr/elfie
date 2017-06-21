#!/bin/bash

FILEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "${FILEDIR}/parameters.sh"

if [[ "$1" != "--only-venv" ]];
then
    >&2 echo "Loading modules.."
    # saved as ~/.lmod.d/<name>, eg. remove the file to redefine
    module restore ${ELFIE_MODULE_COLLECTION} 2&> /dev/null
    if [ $? -eq 0 ];
    then
        >&2 echo "Loaded collection ${ELFIE_MODULE_COLLECTION}"
    else
        for m in "${ELFIE_MODULES[@]}"
        do
            module load ${m}
        done
        module save ${ELFIE_MODULE_COLLECTION}
    fi
    module list
fi

if [[ "$1" != "--only-modules" ]];
then
    >&2 echo "Activating virtualenv.."
    source "${ELFIE_VENV}/bin/activate"
fi

