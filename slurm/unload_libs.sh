#!/bin/bash

HN=`hostname -f`
if [[ $HN == *".triton.aalto.fi" ]];
then
    module purge
    >&2 echo "Unloaded Triton modules"
fi

which deactivate &> /dev/null
resultcode="$?" # 0 -> command exists, 1 -> does not exist
if [[ $resultcode == "0" ]];
then
    >&2 echo "Deactivating virtualenv.."
    deactivate
fi

