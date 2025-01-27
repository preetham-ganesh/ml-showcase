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

# Prepares the Digit Recognizer model for serving.
prepare_model "https://www.dropbox.com/scl/fo/2f8f8bjdzvcq2h317i4fk/ACuxhLpOAaN2xZn8TfeOb3U?rlkey=dy5ths1jxolxw02pfm6ktptrv&st=2b05kbi4&dl=0" \
              "digit_recognizer_v1.0.0" 