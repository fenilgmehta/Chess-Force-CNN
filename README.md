# Chess-Force-CNN
CNN Model based Chess AI

## References
- https://github.com/fenilgmehta/Chess-Force
- https://github.com/fenilgmehta/Chess-Force-Data-Set
- https://stackoverflow.com/questions/31684375/automatically-create-requirements-txt

## Usage

```sh

# DATA_PATH="../../Chess-Force-CNN-Dataset"   # EDIT this so that it points to the directory used for storing all the PGN/CSV/PKL data files used for training/testing/playing
DATA_PATH="."   # EDIT this so that it points to the directory used for storing all the PGN/CSV/PKL data files used for training/testing/playing

PKL_PATH="${DATA_PATH}/04_pkl_data"
PKL_PATH_TRAINED="${DATA_PATH}/04_pkl_data_trained"

SAVED_WEIGHTS_FILE="../../Chess-Force-Models/cnn-mg006-be01588-sn000-ep00016-weight-v001.h5"
MODEL_NAME_PREFIX="cnn"
EPOCHS=8
WEIGHTS_SAVE_PATH="../../Chess-Force-Models"

python step_03b_train.py train                          \
    --gpu=-1                                            \
    --builder_model=6                                   \
    --board_encoder=01588                               \
    --version_number=0                                  \
    --epochs=${EPOCHS}                                  \
    --batch_size=8192                                   \
    --validation_split=0.2                              \
    --input_dir="${PKL_PATH}"                           \
    --move_dir="${PKL_PATH_TRAINED}"                    \
    --file_suffix="pkl"                                 \
    --y_normalizer=5                                    \
    --callback                                          \
    --name_prefix="${MODEL_NAME_PREFIX}"                \
    --saved_weights_file="${SAVED_WEIGHTS_FILE}"        \
    --auto_load_new                                     \
    --weights_save_path="${WEIGHTS_SAVE_PATH}"          \
    && mv "${PKL_PATH_TRAINED}/"*.pkl "${PKL_PATH}"

```