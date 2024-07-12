SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
conda_base=$(conda info --base)
if [ -z "$conda_base" ]; then
    echo "Conda base directory not found. Please ensure conda is installed correctly."
    exit 1
fi
source "${conda_base}/etc/profile.d/conda.sh"

conda activate SPCV

cd $SCRIPT_DIR

# # Get the initial seed from the environment variable or generate a new one
# if [ -z "$DEFAULT_SEED" ]; then
#     SEED=$(python -c 'import random; print(random.randint(0, 2**32 - 1))')
# else
#     SEED=$DEFAULT_SEED
# fi
# # Set fixed seed
SEED=$(python -c 'print(3607073777)')

# Export the seed as an environment variable
export SEED
python A01_fps_mesh.py
python A02_center_scale_unify.py
python SingleGImaker.py
python SqGImaker_10000_adjacent.py
