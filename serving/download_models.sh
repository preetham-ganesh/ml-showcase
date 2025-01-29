#!/bin/bash

# Downloads the model files from the URL, and extracts them to the model directory.
prepare_model() {
    MODEL_URL=$1
    MODEL_NAME=$2

    # Creates the model directory if it does not exist.
    mkdir -p /models/$MODEL_NAME/1

    # Downloads the models, and extracts files from it.
    wget -O /tmp/model_files.zip $MODEL_URL
    unzip /tmp/model_files.zip -d /models/$MODEL_NAME/1
    rm /tmp/model_files.zip
}

# Prepares the Digit Recognizer model v1.0.0 for serving.
prepare_model "https://www.dropbox.com/scl/fo/2f8f8bjdzvcq2h317i4fk/ACuxhLpOAaN2xZn8TfeOb3U?rlkey=dy5ths1jxolxw02pfm6ktptrv&st=2b05kbi4&dl=0" \
              "digit_recognizer_v1.0.0" 

# Prepares the Brain MRI Segmentation - FLAIR Abnormality Classification model v1.2.0 for serving.
prepare_model "https://www.dropbox.com/scl/fo/12be5xjoxt6e4ayd4b5ua/AGEwC65MMfuXnIZHoCNfQHQ?rlkey=4ekks2zxk0jeurn70wezju3f1&st=rb0a9o4g&dl=0" \
              "bms_flair_abnormality_classification_v1.2.0" 

# Prepares the Brain MRI Segmentation - FLAIR Abnormality Segmentation model v1.0.0 for serving.
prepare_model "https://www.dropbox.com/scl/fo/552ixfjpqpxhd02bx0w4i/AH5_W7nEFI05UCTmKd54P4o?rlkey=33f07jcanujedyiz7u3ggl39q&st=zuzbplrj&dl=0" \
              "bms_flair_abnormality_segmentation_v1.0.0" 