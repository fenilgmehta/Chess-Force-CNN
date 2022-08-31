# Chess-Force-CNN

- **Subject** - Foundations of Machine Learning (CS 725)
- **Project** - CNN Model based Chess AI
- **Team Members**
    - 203050054 - Fenil Mehta
    - 203050003 - Aditya Jain
    - 203050115 - Amit Hari
    - 203050120 - Munna Kumar Paswan


## References
1. [Project Report](./CS725%20-%20Chess%20AI%20using%20CNN%20-%20Report.pdf)
2. https://github.com/fenilgmehta/Chess-Force-CNN-Dataset
3. https://github.com/fenilgmehta/Chess-Force-CNN-Models
    * 03 Nov 14:52 👉 [cnn-mg006-be01588-sn005-ep00024-weight-v001.h5](https://github.com/fenilgmehta/Chess-Force-CNN-Models/blob/main/cnn-mg006-be01588-sn005-ep00024-weight-v001.h5)
    * 28 Nov 09:35 👉 [cnn-mg007-be01588-sn006-ep00009-weight-v001.h5](https://github.com/fenilgmehta/Chess-Force-CNN-Models/blob/main/cnn-mg007-be01588-sn006-ep00009-weight-v001.h5)
    * 28 Nov 18:40 👉 [cnn-mg007-be01588-sn007-ep00012-weight-v001.h5](https://github.com/fenilgmehta/Chess-Force-CNN-Models/blob/main/cnn-mg007-be01588-sn007-ep00012-weight-v001.h5)
4. https://github.com/fenilgmehta/Chess-Force
5. https://github.com/fenilgmehta/Chess-Force-Data-Set
6. https://stackoverflow.com/questions/31684375/automatically-create-requirements-txt


## Usage

1. Download the dataset from [Chess-Force-Data-Set](https://github.com/fenilgmehta/Chess-Force-Data-Set)
    - Total board states = 29037071

2. Initialize shell variables
   ```sh
   DATA_PATH='../../Chess-Force-CNN-Dataset'            # EDIT this so that it points to the directory used for storing all the CSV data files used for training/testing/playing
   CSV_SCORES_PATH="${DATA_PATH}/03_csv_score_data"
   CSV_SCORES_PATH_CONVERTED="${DATA_PATH}/03_csv_score_data_converted"
   PKL_PATH="${DATA_PATH}/04_pkl_data"
   PKL_PATH_COMBINED="${DATA_PATH}/04_pkl_data_combined"
   PKL_PATH_TRAINED="${DATA_PATH}/04_pkl_data_trained"
   ```

3. Write your own Board Encoder in [step_02_BoardEncoder.py](./code/step_02_BoardEncoder.py) and Score Normalizer in [step_02_ScoreNormalizer.py](./code/step_02_ScoreNormalizer.py) or reuse the existing ones
   ```sh
   python step_02_preprocess.py --help       # To see how to use the preprocessing module
   python step_02_preprocess.py get_options  # To get the list of parameter you can use for "--board_encoder" and "--score_normalizer"

   # --input_dir           Input data directory
   # --output_dir          Directory to write the generated file
   # --move_dir            Directore to move to the processed files
   # --board_encoder       Board Encoder to use
   # --score_normalizer    Score Normalizer to use
   python step_02_preprocess.py convert_fen_to_pkl_folder    \
       --input_dir="${CSV_SCORES_PATH}"                      \
       --output_dir="${PKL_PATH}"                            \
       --move_dir="${CSV_SCORES_PATH_CONVERTED}"             \
       --board_encoder=01588                                 \
       --score_normalizer=000
   ```

4. Optionally combine the PKL files to make sure whole dataset is used for training instead of having the model the get biased by a part of the dataset
   ```sh
   python step_02_preprocess.py combine_pkls \
       --input_dir="$PKL_PATH"               \
       --output_file="${PKL_PATH_COMBINED}/all_combined.pkl"
   ```

5. Create a model in [step_03a_ffnn.py](./code/step_03a_ffnn.py) inside `class KerasModels` or reuse the existing models
   ```sh
   WEIGHTS_SAVE_PATH="../../Chess-Force-CNN-Models"
   
   MODEL_NAME_PREFIX="cnn"  # Change this prefix based on the model being used
   EPOCHS=4  # Can update this anytime before execution
   
   # Update this whenever required
   SAVED_WEIGHTS_FILE="../../Chess-Force-CNN-Models/cnn-mg007-be01588-sn000-ep00008-weight-v001.h5"
   
   python step_03b_train.py --help              # Will print help message
   python step_03b_train.py get_available_gpus  # Will print available GPU's
   python step_03b_train.py get_options_models  # Will print possible parameters for "--builder_model" argument

   # NOTE: replace "PKL_PATH_COMBINED" in two place below with "PKL_PATH" if step 4 was NOT performed
   python step_03b_train.py train                          \
       --gpu=0                                             \
       --builder_model=7                                   \
       --board_encoder=01588                               \
       --version_number=0                                  \
       --epochs=${EPOCHS}                                  \
       --batch_size=8192                                   \
       --validation_split=0.2                              \
       --generate_model_image                              \
       --input_dir="${PKL_PATH_COMBINED}"                  \
       --move_dir="${PKL_PATH_TRAINED}"                    \
       --file_suffix="pkl"                                 \
       --y_normalizer=7                                    \
       --callback                                          \
       --name_prefix="${MODEL_NAME_PREFIX}"                \
       --saved_weights_file="${SAVED_WEIGHTS_FILE}"        \
       --auto_load_new                                     \
       --weights_save_path="${WEIGHTS_SAVE_PATH}"          \
       && mv "${PKL_PATH_TRAINED}/"*.pkl "${PKL_PATH_COMBINED}"
   ```

6. Play the game
   ```sh
   python step_04_play.py play \
       --game_type=mm                                                                                      \
       --model_weights_file='../../Chess-Force-CNN-Models/cnn-mg007-be01588-sn000-ep00012-weight-v001.h5'  \
       --model_weights_file2='../../Chess-Force-CNN-Models/cnn-mg006-be01588-sn000-ep00024-weight-v001.h5' \
       --analyze_game                                                                                      \
       --delay=0
   ```


