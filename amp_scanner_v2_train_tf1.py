#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
amp_scanner_v2_train_tf1.py
By: Dan Veltri (dan.veltri@gmail.com)
Created: 1.6.2017
Last Modified: 1.9.2025

Takes a protein multi-FASTA file as input and outputs prediction values for each.
Assumes peptides are >= 10 and <= 200 AA in length (AA's longer than 'max_length' are
ignored starting at the FRONT or N-termini by default). Written for Tensorflow v1.x and Keras.

Prediction probabilities > 0.5 predict AMPs and <= 0.5 non-AMPs.

For Usage and help type: 'python mp_scanner_v2_train.py -h'

User should provide training, (optional) validation, and testing FASTA files for AMPs and decoys, respectively.
Ensure 'max_length' is >= to the longest peptide in those files. If you don't want to use validation
data, simply leave it out. You can also use the '-merge' flag to combine the training and validation data together.
Defaults to save 'ascan2_model.h5' model file in the current working directory in HDF5 format - suppress with '-nosave' flag.

NOTE on a multithreaded machine or if using GPU, Tensorflow v1.x can produce stochastic results even if you set a
 random seed. Use the --make_reproducible flag to ensure a deterministic run for a given seed on CPU but note this will
 really slow down the script as it forces a single CPU thread to be used.

While best efforts have been made to ensure the integrity of this script, we take no
responsibility for damages that may result from its use.

This code is released under the GPLv3 license. See the included LICENSE.txt for details.
"""
import argparse
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, matthews_corrcoef, classification_report
from sklearn.utils import shuffle
from Bio import SeqIO
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # turns off tensorflow SSE compile warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning) # ignore numpy future warning in newer versions used

# Fixed model params - ensure these are all valid for your dataset and Tensorflow v1.x and Keras!
max_length = 200
embedding_vector_length = 128
nbf = 64        # No. Conv Filters
flen = 16 	    # Conv Filter length
mpsize = 5      # Max pool size
nlstm = 100     # No. LSTM layers
ndrop = 0.1     # LSTM layer dropout
nbatch = 32     # Fit batch No.
nepochs = 10    # No. training rounds
amino_acids = "XACDEFGHIKLMNPQRSTVWY"
aa2int = dict((c, i) for i, c in enumerate(amino_acids))

def check_format(this_file, valid_aminos, max_len, min_len=10, verbose=True):
    """
        Checks basic format of FASTA file for valid sequence AA characters and length.
        On Entry: this_file is input FASTA file path, valid_aminos is string of valid AA characters,
        max_len and min_len is int of max and min length of sequences allowed, respectively.
        verbose=True prints all errors encountered with the file scan.
        On Exit: Will simply return if no errors found else print errors and system exit unsuccessful.
    """
    if not os.path.exists(this_file):
        print(f"ERROR: FASTA file '{this_file}' not found! Please check path.")
        sys.exit(1)

    errors = []
    try:
        with open(this_file, "r") as fh:
            for line in fh:
                if line.startswith(">") or line.strip().isalpha():
                    continue
                else:
                    errors.append(f"ERROR: Invalid line encountered: '{line.strip()}'")
    except TypeError as e:
        errors.append(f"ERROR: Unable to parse FASTA file '{this_file}':\n{e}\n")
        errors.append("\tIs this file in valid FASTA format? See for details: https://en.wikipedia.org/wiki/FASTA_format\n")

    try:
        for ff in SeqIO.parse(this_file, "fasta"):
            this_seq = str(ff.seq).upper()
            seq_len = len(this_seq)
            invalid_aa = [aa for aa in this_seq if aa not in valid_aminos]
            if invalid_aa:
                errors.append(f"ERROR: In file '{this_file}', invalid sequence provided! '{ff.id}' contains invalid amino acid characters: {', '.join(invalid_aa)}\n")
                errors.append(f"\tPlease fix or remove this sequence, only the 20 classic amino acids or 'X' are accepted.\n")

            if seq_len < min_len or seq_len > max_len:
                errors.append(f"ERROR: In file '{this_file}', sequence length for '{ff.id}' was '{seq_len}' and is out of range!\n")
                errors.append(f"\tMust be >={min_len} and <={max_len} AA in length!\n")

    except ValueError as e:
        errors.append(f"ERROR: Unable to parse FASTA file '{this_file}':\n{e}\n")
        errors.append("\tIs this file in valid FASTA format? See for details: https://en.wikipedia.org/wiki/FASTA_format\n")

    if errors and verbose:
        for msg in errors:
            print(msg)
    if errors:
        sys.exit(1)


def encode_sequences(amp_train_fasta, amp_validate_fasta, amp_test_fasta, decoy_train_fasta, decoy_validate_fasta, decoy_test_fasta, shuffle_seed=123):
    """
        Encodes input AMP and DECOY sequences for use with the Tensorflow model.
        On Entry: each FASTA file is a valid path to a file in valid format. shuffle_seed is used to shuffle train and
         validation data only.
        On Exit: NP arrays of encoded sequences returned as x_train, x_val, x_test for observation splits
         and NP arrays y_train, y_val, y_test for binary classes (1=AMP, 0=DECOY). Train and val data are shuffled.
    """
    x_train, x_val, x_test = [], [], []
    y_train, y_val, y_test = [], [], []

    try:
        print("Encoding training/testing sequences...")
        for s in SeqIO.parse(amp_train_fasta,"fasta"):
            x_train.append([aa2int[aa] for aa in str(s.seq).upper()])
            y_train.append(1)
        if amp_validate_fasta:
            for s in SeqIO.parse(amp_validate_fasta,"fasta"):
                x_val.append([aa2int[aa] for aa in str(s.seq).upper()])
                y_val.append(1)
        for s in SeqIO.parse(amp_test_fasta,"fasta"):
            x_test.append([aa2int[aa] for aa in str(s.seq).upper()])
            y_test.append(1)
        for s in SeqIO.parse(decoy_train_fasta,"fasta"):
            x_train.append([aa2int[aa] for aa in str(s.seq).upper()])
            y_train.append(0)
        x_train = sequence.pad_sequences(x_train, maxlen=max_length)
        x_train, y_train = shuffle(x_train, np.array(y_train), random_state=shuffle_seed)
        if decoy_validate_fasta:
            for s in SeqIO.parse(decoy_validate_fasta,"fasta"):
                x_val.append([aa2int[aa] for aa in str(s.seq).upper()])
                y_val.append(0)
            x_val = sequence.pad_sequences(x_val, maxlen=max_length)
            x_val, y_val = shuffle(x_val, np.array(y_val), random_state=shuffle_seed)
        for s in SeqIO.parse(decoy_test_fasta,"fasta"):
            x_test.append([aa2int[aa] for aa in str(s.seq).upper()])
            y_test.append(0)
        x_test = sequence.pad_sequences(x_test, maxlen=max_length)
    except TypeError as e:
        print(f"ERROR: Failed to properly encode sequences! {e}")
        print("\tPlease review file requirements and FASTA formatting.\n")
        sys.exit(1)

    return x_train, x_val, x_test, np.array(y_train), np.array(y_val), np.array(y_test)


def compile_model(x_train, y_train, x_val, y_val, saved_model_name=None, merge_train_and_val=False):
    """
        Compiles Tensorflow v1.x model on the provided observations and binary classes.
        On Entry: NP arrays of train, test and optional validation data. If string saved_model_name is
          not None, used to save HDF5 model file. If merge_train_and_val is True merges train and val
          data together to train on a merged dataset.
        On Exit: Writes model file if name provided, returns the trained Tensorflow model.
    """
    try:
        print("Compiling model...")
        model = Sequential()
        model.add(Embedding(len(amino_acids), embedding_vector_length, input_length=max_length))
        model.add(Conv1D(filters=nbf, kernel_size=flen, padding="same", activation='relu'))
        model.add(MaxPooling1D(pool_size=mpsize))
        model.add(LSTM(nlstm, use_bias=True, dropout=ndrop, return_sequences=False))#,merge_mode='ave'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    except ValueError as e:
        print(f"ERROR: Failed to compile Tensorflow model! {e}")
        print("\tPlease ensure proper TF parameters for the data are provided!\n")
        sys.exit(1)

    try:
        if merge_train_and_val and y_val.size > 0:
            print("Training on combined train and validation data now...")
            model.fit(np.concatenate((x_train, x_val)), np.concatenate((y_train, y_val)), epochs=nepochs, batch_size=nbatch, verbose=1)
        elif y_val.size > 0:
            print("Training and validating now...")
            model.fit(x_train, y_train, epochs=nepochs, batch_size=nbatch, validation_data=(x_val, y_val), verbose=1)
        else:
            print("Training now...")
            if merge_train_and_val:
                print("WARNING: Ignoring '-merge' flag as validation FASTA was not provided for AMPs and DECOYS!")
            model.fit(x_train, y_train, epochs=nepochs, batch_size=nbatch, verbose=1)
    except ValueError as e:
        print(f"ERROR: Failed to fit Tensorflow model! {e}")
        print("\tEnsure the training/validation data is properly formatted for the model parameters!\n")
        sys.exit(1)

    if saved_model_name is not None:
        try:
            dir_name = os.path.dirname(saved_model_name)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            model.save(saved_model_name)
            print(f"\nSaved model as: {saved_model_name}")
        except IOError as e:
            print(f"ERROR: Could not save model as: {saved_model_name}! {e}")
            sys.exit(1)

    return model


def predict_and_report(x_test, y_test, model):
    """
        Uses trained model and fits on testing data, reporting basic ML performance metrics.
        On Entry: x_test is encoded NP array of testing observations, y_test is NP array of respective classes,
         model is the trained model from compile_model
        On Exit: Prints binary performance metrics on data: TP, TN, FP, FN, Sensitivity, Specificity, Accuracy,
          Matthew's Correlation Coefficient and Precision.
    """
    print("\nGathering Testing Results...")
    try:
        preds = model.predict(x_test)
    except ValueError as e:
        print(f"ERROR: Failed to make model prediction! {e}")
        print("\tIs the testing data in the expected format for the model?\n")
        sys.exit(1)

    try:
        pred_class = np.rint(preds) #round up or down at 0.5
        true_class = np.array(y_test)
        tn, fp, fn, tp = confusion_matrix(true_class,pred_class).ravel()
        roc = roc_auc_score(true_class,preds) * 100.0
        if len(np.unique(true_class)) == 1 or len(np.unique(pred_class)) == 1:
            mcc = 0.0 # undefined
        else:
            mcc = matthews_corrcoef(true_class,pred_class)
        acc = (tp + tn) / (tn + fp + fn + tp + 0.0) * 100.0 if (tn + fp + fn + tp) > 0 else 0.0
        sens = tp / (tp + fn + 0.0) * 100.0 if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp + 0.0) * 100.0 if (tn + fp) > 0 else 0.0
        prec = tp / (tp + fp + 0.0) * 100.0 if (tp + fp) > 0 else 0.0

        print("\nTP\tTN\tFP\tFN\tSens\tSpec\tAcc\tMCC\tauROC\tPrec")
        print(f"{tp}\t{tn}\t{fp}\t{fn}\t{sens:.2f}\t{spec:.2f}\t{acc:.2f}\t{mcc:.4f}\t{roc:.2f}\t{prec:.2f}")
    except ValueError as e:
        print(f"ERROR: Evaluating model performance! {e}")
        print("\tDo both X and y test have the expected format and the same number of observations?\n")

def main():

    parser = argparse.ArgumentParser(
        prog="amp_scanner_v2_train_tf1.py",
        description="Script to train and evaluate AMP Scanner v2 Tensorflow model with AMP and DECOY FASTA datasets.")

    parser.add_argument("-atr",
                        "--amp_train_fasta",
                        required=True,
                        type=str,
                        help="Path to REQUIRED AMP training FASTA file.")

    parser.add_argument("-ate",
                        "--amp_test_fasta",
                        required=True,
                        type=str,
                        help="Path to REQUIRED AMP test FASTA file.")

    parser.add_argument("-ava",
                        "--amp_validate_fasta",
                        required=False,
                        type=str,
                        default=None,
                        help="Path to OPTIONAL AMP validation FASTA file (default: None - no validation set used).")

    parser.add_argument("-dtr",
                        "--decoy_train_fasta",
                        required=True,
                        type=str,
                        help="Path to REQUIRED decoy training FASTA file.")

    parser.add_argument("-dte",
                        "--decoy_test_fasta",
                        required=True,
                        type=str,
                        help="Path to REQUIRED decoy test FASTA file.")

    parser.add_argument("-dva",
                        "--decoy_validate_fasta",
                        required=False,
                        type=str,
                        default=None,
                        help="Path to OPTIONAL decoy validation FASTA file (default: None - no validation set used).")

    parser.add_argument("-o",
                        "-out",
                        "--output_model_name",
                        required=False,
                        type=str,
                        default='ascan2_model.h5',
                        help="Optional output filename for the saved HDF5 .h5 model (default: ascan2_model.h5 in current directory).")

    parser.add_argument("-n",
                        "-nosave",
                        "--skip_output_model",
                        required=False,
                        action="store_true",
                        default=False,
                        help="OPTIONAL flag to skip saving output model - print out model evaluation only.")

    parser.add_argument("-m",
                        "-merge",
                        "--merge_train_and_val",
                        required=False,
                        action="store_true",
                        default=False,
                        help="OPTIONAL flag to merge validation data with training data for combined training.")

    parser.add_argument("-s",
                        "-seed",
                        "--shuffle_seed",
                        required=False,
                        type=int,
                        default=123,
                        help="Optional seed for random shuffling of training and validation data partitions (default: 123).")

    parser.add_argument("-r",
                        "-reprod",
                        "--make_reproducible",
                        required=False,
                        action="store_true",
                        default=False,
                        help="Optional setting to make training runs reproducible (for a given seed) with TFv1. NOTE: This will force CPU to use a single thread and performance will be SLOW!")

    args = parser.parse_args()

    # See comments or README for information on reproducibility - this will slow down the code and use a single CPU thread!
    if args.make_reproducible:
        config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1, device_count={"GPU": 0})
        sess = tf.Session(config=config)
    tf.set_random_seed(args.shuffle_seed)
    np.random.seed(args.shuffle_seed)

    if args.skip_output_model:
        print("Running in Evaluation Mode: No output model file will be saved!")
        args.output_model_name = None

    file_list = [args.amp_train_fasta, args.amp_test_fasta, args.decoy_train_fasta, args.decoy_test_fasta]

    if args.amp_validate_fasta and args.decoy_validate_fasta:
        file_list += [args.decoy_validate_fasta, args.decoy_validate_fasta]
    elif args.amp_validate_fasta or args.decoy_validate_fasta:
        print("ERROR: A single validation FASTA was provided!")
        print("\tIf validation data used, it must be provided for BOTH the AMP and DECOY classes!\n")
        sys.exit(1)

    for f in file_list:
        check_format(f, amino_acids, max_length)

    x_train, x_val, x_test, y_train, y_val, y_test = encode_sequences(args.amp_train_fasta,
                                                                      args.amp_validate_fasta,
                                                                      args.amp_test_fasta,
                                                                      args.decoy_train_fasta,
                                                                      args.decoy_validate_fasta,
                                                                      args.decoy_test_fasta,
                                                                      args.shuffle_seed)

    model = compile_model(x_train,
                          y_train,
                          x_val,
                          y_val,
                          args.output_model_name,
                          args.merge_train_and_val)

    predict_and_report(x_test, y_test, model)

if __name__ == "__main__":
    main()

# END PROGRAM
