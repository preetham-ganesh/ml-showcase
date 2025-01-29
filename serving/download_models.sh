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
prepare_model "https://www.dropbox.com/scl/fo/3nuexl16h3muaexzdcspd/AGDcODtLO2v43iWU2Y27wEE?rlkey=ad1kbd6ybpcbpfxvocuiwac3n&st=c1fjjb8g&dl=0" \
              "bms_flair_abnormality_classification_v1.2.0" 

# Prepares the Brain MRI Segmentation - FLAIR Abnormality Segmentation model v1.0.0 for serving.
prepare_model "https://www.dropbox.com/scl/fo/k7kxvtt0soohbt4p8h9xi/AI4ZSPiKG-mwtesBuAHVyhM?rlkey=byfx2wawopv4ck7itwr3y521l&st=nlueuec5&dl=0" \
              "bms_flair_abnormality_segmentation_v1.0.0" 