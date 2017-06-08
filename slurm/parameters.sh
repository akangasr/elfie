# environment setup
ELFIE_ROOT="${HOME}/work2"
ELFIE_SOURCE="${ELFIE_ROOT}/elfi2"
ELFIE_RESULTS="${ELFIE_ROOT}/results"
ELFIE_VENV="${ELFIE_ROOT}/.venv"
ELFIE_MODULES=(
    "Python/3.5.1-goolf-triton-2016a"
    "OpenMPI/2.0.2-GCC-6.3.0-2.27"
    )
ELFIE_REQUIREMENTS=(
    "${ELFIE_ROOT}/elfi-experiment/requirements.txt"
    )
ELFIE_GIT_REPOS=(
    "${ELFIE_ROOT}/elfi2"
    "${ELFIE_ROOT}/elfi-experiment"
    )
