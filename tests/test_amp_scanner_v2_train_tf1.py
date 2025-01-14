#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import os
import unittest
from distutils.version import LooseVersion # deprecated python >= 3.10
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, matthews_corrcoef, classification_report
from Bio import SeqIO
from amp_scanner_v2_train_tf1 import check_format, encode_sequences, predict_and_report, compile_model
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.embeddings import Embedding
np.random.seed(123)

class TestAScan2(unittest.TestCase):

    def setUp(self):
        self.loaded_model = None
        self.tf1_model = "../trained-models/020419_FULL_MODEL.h5"
        if not os.path.exists(self.tf1_model):
            self.fail(f"2019+ TFv1 model not where it is expected: {self.tf1_model}")
        self.orig_model = "../trained-models/OriginalPaper_081917_FULL_MODEL.h5"
        if not os.path.exists(self.tf1_model):
            self.fail(f"Original paper TFv1 model not where it is expected: {self.tf1_model}")
        self.valid_fasta = "tests/sample_valid.fasta"
        self.invalid_fasta = "tests/sample_invalid.fasta"
        self.fake_sequences = [
            ">Seq_OK\nLLGDFFRKAKEKIGKESKRIVQRIKDFLRNLVPRTES\n",
            ">Seq_OK_X\nLLGDFFRKAKEKIGXESKRIVQRIKDFLRNLVPRTES\n",
            ">Seq_BAD_Short\nLLGDFF\n"
            ">Seq_BAD_Long\nMKKPSKKSEIEFCTVCRFHHDQGSRHKYFPRHKSSLSSLLDRFRSKIADVRFFLKNPSVLRPQEQSQNRVWCVFCDEDIVELGSSFACSKAINHFASSDHLKNIKQFLSKNGPAMDCIDEFRISEADVAKWEKKCQSFGNEDASFEGSCGQLSGTSNDIHTKLAFETMDRIKKVPAHHINSYKSNDVMPLQYNTNEYQISLSEI\n"
            ">Seq_BAD_InvalidAA\nLLGDFFRKAZEKIGKESKRIVQRIKDFLRNLVPRTES\n"
            "$Seq_BAD_Definition\nLLGDFFRKAKEKIGKESKRIVQRIKDFLRNLVPRTES\n",
        ]
        with open(self.valid_fasta, "w") as fh:
            fh.write("".join(self.fake_sequences[:2]))
        with open(self.invalid_fasta, "w") as fh:
            fh.write("".join(self.fake_sequences[2:]))
        self.test_amps = "tests/test_amps.fasta"
        self.test_amp_seqs = [
            ">AP00695\nFLPILGKLLSGIL\n",
            ">AP00658\nFLPLVGKILSGLI\n",
            ">AP00324\nGVGDLIRKAVSVIKNIV\n",
        ]
        with open(self.test_amps, "w") as fh:
            fh.write("".join(self.test_amp_seqs))
        self.test_decoys = "tests/test_decoys.fasta"
        self.test_decoy_seqs = [
            ">UniRef50_Q9BRA0\nNVILGSAQEFLKPSD\n",
            ">UniRef50_Q54JQ2\nPKENPFLPIDTTIKAPQDHSIHIPKEVYNNNGVKVYHSLDHRFNSPKARVNIRFELTSYGNNQSMVM\n",
            ">UniRef50_E0RTD9\nDGILTRGGTILGTSREKPFKPDPGEKDSEAGSRKVEAIIENYHK\n",
        ]
        with open(self.test_decoys, "w") as fh:
            fh.write("".join(self.test_decoy_seqs))
        self.amino_acids = "XACDEFGHIKLMNPQRSTVWY"
        self.aa2int = dict((c, i) for i, c in enumerate(self.amino_acids))
        self.max_length = 200
        self.min_length = 10

    def tearDown(self):
        if os.path.exists(self.valid_fasta):
            os.remove(self.valid_fasta)
        if os.path.exists(self.invalid_fasta):
            os.remove(self.invalid_fasta)
        if os.path.exists(self.test_amps):
            os.remove(self.test_amps)
        if os.path.exists(self.test_decoys):
            os.remove(self.test_decoys)
        if self.loaded_model:
            del self.loaded_model
        if os.path.exists("test/model.h5"):
            os.remove("test/model.h5")

    def test_check_format_valid_file(self):
        try:
            check_format(self.valid_fasta, self.amino_acids, self.max_length, self.min_length, verbose=False)
        except SystemExit as e:
            self.fail(f"File format test failed: check_format raised exception unexpectedly with a valid file! {e}")

    def test_check_format_invalid_file(self):
        with self.assertRaises(SystemExit):
            check_format(self.invalid_fasta, self.amino_acids, self.max_length, self.min_length, verbose=False)

    def test_biopython_input(self):
        try:
            for ff in SeqIO.parse(self.valid_fasta, "fasta"):
                seq = ff.seq
        except SystemExit as e:
            self.fail(f"Biopython SeqIO failed to read valid FASTA: check biopython is installed correctly! {e}")

    def test_encode_sequences(self):
        x_train, x_val, x_test, y_train, y_val, y_test = encode_sequences(
            self.test_amps, self.test_amps, self.test_amps,
            self.test_decoys, self.test_decoys, self.test_decoys
        )
        self.assertEqual(len(x_train), len(y_train))
        self.assertEqual(x_train.shape[1], self.max_length)  # Padded to max_length

    def test_compile_model(self):
        x_train, x_val, x_test, y_train, y_val, y_test = encode_sequences(
            self.test_amps, self.test_amps, self.test_amps,
            self.test_decoys, self.test_decoys, self.test_decoys
        )
        test_model = compile_model(x_train, y_train, x_val, y_val, saved_model_name=None, merge_train_and_val=True)
        self.assertIsInstance(test_model, Sequential, "compile_model did not return a Keras Sequential model.")

    def test_compile_save_output(self):
        x_train, x_val, x_test, y_train, y_val, y_test = encode_sequences(
            self.test_amps, self.test_amps, self.test_amps,
            self.test_decoys, self.test_decoys, self.test_decoys
        )
        test_model = compile_model(x_train, y_train, x_val, y_val, saved_model_name="test/model.h5", merge_train_and_val=False)
        self.assertIsInstance(test_model, Sequential, "compile_model did not return a merged Keras Sequential model.")
        self.assertTrue(os.path.exists("test/model.h5"), f"Model file was not created at 'test/model.h5'")

    def test_predict_and_report(self):
        x_test = [[self.aa2int[aa] for aa in "FLPILGKLLSGIL"], [self.aa2int[aa] for aa in "NVILGSAQEFLKPSD"],
                  [self.aa2int[aa] for aa in "GVGDLIRKAVSVIKNIV"],
                  [self.aa2int[aa] for aa in "DGILTRGGTILGTSREKPFKPDPGEKDSEAGSRKVEAIIENYHK"]]
        x_test = sequence.pad_sequences(x_test, maxlen=self.max_length)
        y_test = np.array([1,0,1,0])

        test_model = Sequential()
        test_model.add(Embedding(21, 128, input_length=self.max_length))
        test_model.add(Conv1D(64, 16, padding="same", activation='relu'))
        test_model.add(MaxPooling1D(pool_size=5))
        test_model.add(LSTM(100, return_sequences=False))
        test_model.add(Dense(1, activation='sigmoid'))
        test_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        try:
            test_model.predict(x_test)
        except ValueError as e:
            self.fail(f"Unable to predict with test_model! {e}")

        try:
            predict_and_report(x_test, y_test, test_model)
        except ValueError as e:
            self.fail(f"predict_and_report raised an exception unexpectedly. {e}")

    def test_sklearn_evaluation(self):
        preds = np.array([0.1, 0.99, 0.23, 0.4, 0.5, 0.51, 0.7, 0.12, 0.1, 0.99])
        true_class = np.array([0,0,0,0,0,1,1,1,1,1])
        pred_class = np.rint(preds) # Round up or down at 0.5

        try:
            tn, fp, fn, tp = confusion_matrix(true_class,pred_class).ravel() # expect 4, 1, 2, 3
            if tn != 4 or fp != 1 or fn != 2 or tp != 3:
                raise ValueError("Confusion matrix returned incorrect values!")
        except ValueError as e:
            self.fail(f"sklearn confusion_matrix raised an exception {e}. Is scikit-learn installed correctly?")

        try:
            roc = roc_auc_score(true_class,preds) * 100.0
            if np.round(roc) != 56.0:
                raise ValueError("auROC returned incorrect value!")
        except ValueError as e:
            self.fail(f"sklearn roc_auc_score raised an exception {e}. Is scikit-learn installed correctly?")

        try:
            mcc = matthews_corrcoef(true_class,pred_class)
            if np.round(mcc,4) != 0.4082:
                raise ValueError("MCC returned incorrect value!")
        except ValueError as e:
            self.fail(f"sklearn matthews_corrcoef raised an exception {e}. Is scikit-learn installed correctly?")

    def test_tensorflow_version_and_model_compatibility(self):
        tf_version = tf.__version__
        x_test = [[self.aa2int[aa] for aa in "FLPILGKLLSGIL"], [self.aa2int[aa] for aa in "NVILGSAQEFLKPSD"],
                  [self.aa2int[aa] for aa in "GVGDLIRKAVSVIKNIV"],
                  [self.aa2int[aa] for aa in "DGILTRGGTILGTSREKPFKPDPGEKDSEAGSRKVEAIIENYHK"]]
        x_test = sequence.pad_sequences(x_test, maxlen=self.max_length)
        y_test = np.array([1,0,1,0])

        # Check compatibility with each model
        if LooseVersion(tf_version) >= LooseVersion("2.0"):
            self.fail(f"This environment has TFv2 installed {tf_version} - this version of the script requires TFv1!")

        elif LooseVersion(tf_version) >= LooseVersion("1.12"):
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                # This TFv1 version is compatible with the 2019+ TFv1 models.
                try:
                    self.loaded_model = load_model(self.tf1_model)
                    predict_and_report(x_test, y_test, self.loaded_model)
                except KeyError as e:
                    print(f"Failed to load 2019+ later TFv1 model: {e}")
                try:
                    self.loaded_model = load_model(self.orig_model)
                    predict_and_report(x_test, y_test, self.loaded_model)
                except KeyError:
                    print(f"\n\nWARNING: Environment (TF:{tf_version}) is configured to run the 2019+ TFv1 models, but will not work with the original paper version model (requires TF 1.2.1). This is expected.\n\n")

        elif LooseVersion(tf_version) == LooseVersion("1.2.1"):
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                # This TFv1 version is compatible with the original paper version model.
                try:
                    self.loaded_model = load_model(self.orig_model)
                    predict_and_report(x_test, y_test, self.loaded_model)
                except KeyError as e:
                    print(f"Failed to load original paper TFv1 model: {e}")
                try:
                    self.loaded_model = load_model(self.tf1_model)
                    predict_and_report(x_test, y_test, self.loaded_model)
                except KeyError:
                    print(f"\n\nWARNING: Environment (TF:{tf_version}) is configured to run the original paper version of the model, but will not work with the 2019 or later TFv1 models (requires TF 1.12). This is expected.\n\n")

        else:
            self.fail("Environment (TF:{tf_version}) loading tests failed! This environment is not configured correctly to work with any of the TFv1 models. Please refer to the README and requirements files for more information!")