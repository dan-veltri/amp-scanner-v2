#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
amp_scanner_v2_predict_tf1.py
By: Dan Veltri (dan.veltri@gmail.com)
Created: 1.6.2017
Last Modified 12.20.2024

Description: Makes AMP predictions given a FASTA file and trained Tensorflow v1 and Keras .h5 model as input.

For usage run: python amp_scanner_v2_predict_tf1.py -h

Proteins in <query.fasta> must be 20 classic AAs or X (treated as a padding character).
Keras and TensorFlow version of <model.h5> HDF5 model should be same as the current environment.

Output:	Saves to the current working directory -
		<query_basename>_AMP_Candidates.fasta - FASTA file of peptides predicted as AMPs
		<query_basename>_Prediction_Summary.csv - CSV file of prediction results

Prediction probabilities > 0.5 signify AMPs, <= 0.5 signify non-AMPs.

Citation: D. Veltri, U. Kamath and A. Shehu (2018) Bioinformatics, 34(16):2740–2747.

While best efforts have been made to ensure the integrity of this script, we take no
responsibility for damages that may result from its use.

This code is released under the GPLv3 license. See the included LICENSE.txt for details.
"""
from time import gmtime, strftime
import argparse
from Bio import SeqIO
from keras.models import load_model
from keras.preprocessing import sequence
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' # turns off tensorflow SSE compile warnings for CPU

amino_acids = "XACDEFGHIKLMNPQRSTVWY"

def check_and_encode_format(this_file, valid_aminos, max_len, min_len=10):
    """
        Checks basic format of FASTA file for valid sequence AA characters and length and encodes sequences.
        On Entry: this_file is input FASTA file path, valid_aminos is string of valid AA characters,
        max_len and min_len is int of max and min length of sequences allowed, respectively.
        On Exit: Will simply return if no errors found else print errors and system exit unsuccessful.
    """
    print("Encoding sequences...")
    aa2int = dict((c, i) for i, c in enumerate(valid_aminos))
    x, ids, seqs, errors = [], [], [], []
    warnings_found = False
    try:
        with open(this_file, "r") as fh:
            for line in fh:
                if line.startswith(">") or line.strip().isalpha():
                    continue
                else:
                    errors.append(f"Invalid line encountered: '{line.strip()}'")
    except Exception as e:
        errors.append(f"ERROR: Unable to parse FASTA file '{this_file}': {e}")
        errors.append("\tIs this file in valid FASTA format? See for details: https://en.wikipedia.org/wiki/FASTA_format")

    try:
        for ff in SeqIO.parse(this_file, "fasta"):
            this_seq = str(ff.seq).upper()
            seq_len = len(this_seq)
            this_id = str(ff.id)
            invalid_aa = [aa for aa in this_seq if aa not in valid_aminos]
            if invalid_aa:
                errors.append(f"ERROR: In file '{this_file}', invalid sequence provided! '{this_id}' contains invalid amino acid characters: {', '.join(invalid_aa)}")
                errors.append(f"\tPlease fix or remove this sequence, only these characters are accepted: X,A,C,D,E,F,G,H,I,K,L,M,N,P,Q,R,S,T,V,W,Y.")
            else:
                seqs.append(this_seq)
                x.append([aa2int[aa] for aa in this_seq])
            if seq_len < min_len:
                errors.append(f"ERROR: In file '{this_file}', sequence length for '{this_id}' was '{seq_len}'")
                errors.append(f"\tPlease remove this sequence, all MUST be >= {min_len}!")
            if seq_len > max_len:
                print(f"WARNING: In file '{this_file}', sequence length for '{this_id}' was '{seq_len}' - less than the RECOMMENDED max length of {max_len}! Appending '*' to the sequence ID to denote this!")
                this_id += '*'
                warnings_found = True
            ids.append(this_id)
    except ValueError as e:
        errors.append(f"ERROR: Unable to parse FASTA file '{this_file}': {e}")
        errors.append("\tIs this file in valid FASTA format? See for details: https://en.wikipedia.org/wiki/FASTA_format")

    if errors:
        for msg in errors:
            print(msg)
        sys.exit(1)

    return ids, seqs, sequence.pad_sequences(x, maxlen=max_len), warnings_found

def predict_amps(x_test, model):
    print("Making predictions...")
    try:
        loaded_model = load_model(model)
    except Exception as e:
        print(f"ERROR: Unable to load model file {model}: {e}")
        print(f"\tPlease check if this is a valid model file for the specific Tensorflow version required and in HDF5 format!")
        sys.exit(1)

    try:
        preds = loaded_model.predict(x_test)
    except Exception as e:
        print(f"ERROR: Failed to make predictions with loaded model: {e}")
        if 'expected embedding_1_input to have shape' in str(e):
            print(f"\tEnsure the 'max_length' argument matches the length that was used to train the model file {model}!")
        sys.exit(1)

    return preds

def save_output(ids, seqs, preds, thrsh=0.5, candidates_filename='AMP_Candidates.fasta', predictions_filename='Prediction_Summary.csv'):
    print("Saving output...")
    fout_candidates, fout_predictions = None, None
    try:
        fout_candidates = open(candidates_filename,'w')
        fout_predictions = open(predictions_filename,'w')
        fout_predictions.write("SeqID,Prediction_Class,Prediction_Probability,Sequence\n")
        for i, pred in enumerate(preds):
            if pred[0] > thrsh:
                fout_predictions.write(f"{ids[i]},AMP,{pred[0]:.4f},{seqs[i]}\n")
                fout_candidates.write(f">{ids[i]}\n{seqs[i]}\n")
            else:
                fout_predictions.write(f"{ids[i]},Non-AMP,{pred[0]:.4f},{seqs[i]}\n")
        print(f"Saved files: {candidates_filename} and {predictions_filename}")
    except IOError as e:
        print(f"ERROR: Failed to save output files '{candidates_filename}' or '{predictions_filename}': {e}")
        sys.exit(1)
    finally:
        if fout_candidates:
            fout_candidates.close()
        if fout_predictions:
            fout_predictions.close()

def main():

    parser = argparse.ArgumentParser(
        prog="amp_scanner_v2_predict_tf1.py",
        description="Script to predict AMPs on a query FASTA file using an AMP Scanner v2 Tensorflow model.")

    parser.add_argument("-f",
                        "-q",
                        "-fasta",
                        "-query",
                        "--query_fasta",
                        required=True,
                        type=str,
                        help="REQUIRED: Path to query protein FASTA file (see README for acceptable amino acid characters and protein sequence lengths).")

    parser.add_argument("-m",
                        "-model",
                        "--model_file",
                        required=True,
                        type=str,
                        help="REQUIRED: Path to AMP Scanner v2 Tensorflow .h5 model file in HDF5 format.")

    parser.add_argument("-c",
                        "-candidates",
                        "--candidate_amp_fasta_output",
                        required=False,
                        type=str,
                        default="AMP_Candidates.fasta",
                        help="OPTIONAL: Path and filename for output AMP candidates FASTA file (default: <query_fasta_basename>_AMP_Candidates.fasta)")

    parser.add_argument("-p",
                        "-preds",
                        "--predictions_csv_output",
                        required=False,
                        type=str,
                        default="AMP_Predictions.csv",
                        help="OPTIONAL: Path and filename for output AMP predictions CSV file (default: <query_fasta_basename>_AMP_Predictions.csv)")

    parser.add_argument("-l",
                        "--max_length",
                        required=False,
                        type=int,
                        default=200,
                        help="OPTIONAL: Max length expected in query sequences. The <model_file> must have been trained on this same size! (default: 200, note AMP Scanner v2 models were not trained beyond this, change with caution!)")

    parser.add_argument("-t",
                        "-thrsh",
                        "--threshold_cutoff",
                        required=False,
                        type=float,
                        default=0.5,
                        help="OPTIONAL: Threshold cutoff between 0-1, GREATER than this is predicted as an AMP (default: >0.5)")

    parser.add_argument("-ns",
                        "-nosave",
                        "--skip_output_files",
                        required=False,
                        action="store_true",
                        default=False,
                        help="OPTIONAL flag to skip saving output files - print predictions to STDOUT instead.")

    parser.add_argument("-nt",
                        "-notime",
                        "--skip_timing",
                        required=False,
                        action="store_true",
                        default=False,
                        help="OPTIONAL flag to skip reporting time results to STDOUT.")

    args = parser.parse_args()

    if not args.skip_timing:
        print("STARTING JOB: " + strftime("%Y-%m-%d %H:%M:%S", gmtime()))

    if not os.path.exists(args.query_fasta):
        print(f"ERROR: FASTA file '{args.query_fasta}' not found! Please check path.")
        sys.exit(1)

    if not os.path.exists(args.model_file):
        print(f"ERROR: model file '{args.model_file}' not found! Please check path.")
        sys.exit(1)

    if args.threshold_cutoff <= 0.0 or args.threshold_cutoff >= 1.0:
        print(f"ERROR: Threshold cutoff must be between 0.0 and 1.0, you provided: {args.threshold_cutoff}")
        sys.exit(1)

    ids, seqs, x_test, warnings_found = check_and_encode_format(args.query_fasta, amino_acids, args.max_length)
    preds = predict_amps(x_test, args.model_file)

    if args.skip_output_files:
        print("ID,Prediction_Class,Prediction_Probability,Sequence")
        for i, pred in enumerate(preds):
            if pred[0] > args.threshold_cutoff:
                print(f"{ids[i]},AMP,{pred[0]:.4f},{seqs[i]}")
            else:
                print(f"{ids[i]},Non-AMP,{pred[0]:.4f},{seqs[i]}")
    else:
        basefile = os.path.basename(args.query_fasta)
        basename = os.path.splitext(basefile)[0]
        candidates_filename = os.path.join(os.getcwd(), f"{basename}_{args.candidate_amp_fasta_output}")
        predictions_filename = os.path.join(os.getcwd(), f"{basename}_{args.predictions_csv_output}")
        save_output(ids, seqs, preds, args.threshold_cutoff, candidates_filename, predictions_filename)

    if warnings_found:
        print(f"\nNOTE: A '*' was appended to IDs for sequences longer than the max length setting ({args.max_length} amino acids). These results may be unreliable!")
    if args.threshold_cutoff != 0.5:
        print(f"\nNOTE: The prediction threshold was changed from the default value (0.5) to be >{args.threshold_cutoff} is predicted to be an AMP.")
    if args.max_length != 200:
        print(f"\nNOTE: The max length was changed from the default value (200) to be {args.max_length} amino acids.")
    if not args.skip_timing:
        print("JOB FINISHED: " + strftime("%Y-%m-%d %H:%M:%S", gmtime()))


if __name__ == "__main__":
    main()

# END PROGRAM