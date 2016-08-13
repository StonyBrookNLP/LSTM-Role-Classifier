import json
import os
import numpy as np
import pandas as pd

from keras.callbacks import History
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding, TimeDistributed, LSTM, Merge
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.preprocessing.sequence import pad_sequences

import preprocess

y_map = {"PADDING": 0, "A0": 1, "A1": 2, "A2": 3, "A3": 4, "NONE": 5}

one_hot_y = {0: [1, 0, 0, 0, 0, 0],
             1: [0, 1, 0, 0, 0, 0],
             2: [0, 0, 1, 0, 0, 0],
             3: [0, 0, 0, 1, 0, 0],
             4: [0, 0, 0, 0, 1, 0],
             5: [0, 0, 0, 0, 0, 1]}

# process in each of the five folds
f1_processes = ["convection", "chemical_reaction", "acceleration", "meiosis",
                "refraction", "photosynthesis", "regeneration",
                "sexual_reproduction", "respiration", "shedding", "orbit",
                "carbon_cycle"]
f2_processes = ["dripping", "physical_change", "reflection", "absorption",
                "pollination", "mitosis", "nitrogen_cycle", "germination",
                "cross-pollination", "conservation", "combustion", "boiling",
                "erosion", "metamorphosis"]
f3_processes = ["chemosynthesis", "osmosis", "diffusion", "terracing",
                "crossbreeding", "grafting", "strip_farming", "melting",
                "decomposition", "asexual_reproduction", "chemical_change",
                "evaporation"]
f4_processes = ["molting", "fertilization", "nitrogen_fixation", "conduction",
                "mimicry", "condensation", "digestion", "camouflage",
                "crop_rotation", "sublimation"]
f5_processes = ["breathing", "weathering", "evolution", "hibernation",
                "friction", "reproduction"]

MAX_SEQUENCE_LENGTH = 50
BATCH_SIZE = 1
NUM_LABELS = 6  # one-hot vector size
EMBEDDING_DIM = 100
NUM_EPOCHS = 1
LSTM_DEPTH = 1


def main():
    # pass cleaned dataset file containing sentences with non-overlapping
    # argument spans
    d = preprocess.Dataset("cleaned_input_data.tsv")

    # create output directory if needed
    for fold in ['1', '2', '3', '4', '5']:
        dir_name = 'out_data/fold-' + fold + '/test'
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        dir_name = 'out_data/fold-' + fold + '/train'
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

    # vocabulary size
    VOCABULARY_SIZE = len(d.get_vocabulary())

    all_processes = f1_processes + f2_processes + f3_processes + f4_processes + f5_processes

    for f_num, f_list in enumerate([f1_processes, f2_processes, f3_processes, f4_processes, f5_processes]):
        pass

        # fold f_num as test and rest of the folds as train
        test_processes = f_list
        train_processes = list(set(all_processes) - set(f_list))

        # Process data and create train and test matrices for the model
        test_map_data = []
        s_embedings = {}
        X_train_raw = []
        Y_train_raw = []
        X_test_raw = []
        Y_test_raw = []
        args_bools = []
        for process_id, process in enumerate(d):
            for sentence in process:
                sent_X = []
                sent_Y = []
                args_bool = []
                for token in sentence:
                    sent_X.append(token.get_token_id())
                    sent_Y.append(y_map[token.get_label()])
                    args_bool.append(token.arg)
                    if process.name in train_processes:
                        pass
                    else:
                        test_map_data.append([sentence.sentence_id, token.get_token(), token.get_label()])
                    if token.get_token_id() not in s_embedings:
                        s_embedings[token.get_token_id()] = token.get_embedding()
                if process.name in train_processes:
                    X_train_raw.append(sent_X)
                    Y_train_raw.append(sent_Y)
                else:
                    args_bools.append(args_bool)
                    X_test_raw.append(sent_X)
                    Y_test_raw.append(sent_Y)

        # pad sequences to run in batch mode
        args_bools = pad_sequences(args_bools, maxlen=MAX_SEQUENCE_LENGTH)
        X_train = pad_sequences(X_train_raw, maxlen=MAX_SEQUENCE_LENGTH)
        Y_train_raw = pad_sequences(Y_train_raw, maxlen=MAX_SEQUENCE_LENGTH)
        Y_train = [[one_hot_y[entry] for entry in y] for y in Y_train_raw]

        X_test = pad_sequences(X_test_raw, maxlen=MAX_SEQUENCE_LENGTH)
        Y_test_raw = pad_sequences(Y_test_raw, maxlen=MAX_SEQUENCE_LENGTH)
        Y_test = [[one_hot_y[entry] for entry in y] for y in Y_test_raw]

        Y_train = np.asarray(Y_train, dtype='int32')
        Y_test = np.asarray(Y_test, dtype='int32')

        # create embedding matrix for initialization using
        # pre-trained embeddings
        # embedding_map = {}
        embedding_matrix = np.zeros((VOCABULARY_SIZE + 1, EMBEDDING_DIM))
        for word_id, embedding_vector in s_embedings.items():
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[word_id] = embedding_vector
                # embedding_map[id_to_word[word_id]] = embedding_vector

        # pickle.dump(embedding_map, open("utils/embedding_map", "wb"))

        # define the model

        # LSTM Model Start  ----------------------------------------------------
        model = Sequential()
        model.add(Embedding(VOCABULARY_SIZE + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False))
        # hidden LSTM layer(s)
        for i in range(LSTM_DEPTH):
            model.add(LSTM(EMBEDDING_DIM, return_sequences=True))
            model.add(Dropout(0.5))
        model.add(TimeDistributed(Dense(NUM_LABELS)))
        model.add(Activation('softmax'))

        rms = RMSprop()
        model.compile(loss='categorical_crossentropy',
                      optimizer=rms,
                      metrics=["accuracy"])

        history = History()
        print "\nRunning Train:"
        model.fit(X_train, Y_train,
                  batch_size=BATCH_SIZE,
                  nb_epoch=NUM_EPOCHS,
                  callbacks=[history])

        print "\nLOSS:"
        for epoch, val in enumerate(history.history['loss']):
            print epoch+1, val

        print "\nRunning Test:"
        res = model.predict_classes(X_test)
        res_probs = model.predict_proba(X_test, verbose=0)
        # LSTM Model End  ------------------------------------------------------

        # B-LSTM Model Start  --------------------------------------------------
        # left = Sequential()
        # left.add(Embedding(VOCABULARY_SIZE + 1,
        #                    EMBEDDING_DIM,
        #                    weights=[embedding_matrix],
        #                    input_length=MAX_SEQUENCE_LENGTH,
        #                    trainable=False))
        # for i in range(LSTM_DEPTH):
        #     left.add(LSTM(EMBEDDING_DIM, return_sequences=True))
        #     left.add(Dropout(0.5))

        # right = Sequential()
        # right.add(Embedding(VOCABULARY_SIZE + 1,
        #                     EMBEDDING_DIM,
        #                     weights=[embedding_matrix],
        #                     input_length=MAX_SEQUENCE_LENGTH,
        #                     trainable=False))
        # for i in range(LSTM_DEPTH):
        #     right.add(LSTM(EMBEDDING_DIM, return_sequences=True))
        #     right.add(Dropout(0.5))

        # model = Sequential()
        # model.add(Merge([left, right], mode='concat'))
        # model.add(TimeDistributed(Dense(NUM_LABELS)))
        # model.add(Activation('softmax'))

        # rms = RMSprop()
        # model.compile(loss='categorical_crossentropy',
        #               optimizer=rms,
        #               metrics=["accuracy"])

        # history = History()
        # print "\nRunning Train:"
        # model.fit([X_train, X_train], Y_train,
        #           batch_size=BATCH_SIZE,
        #           nb_epoch=NUM_EPOCHS,
        #           callbacks=[history])

        # print "\nLOSS:"
        # for epoch, val in enumerate(history.history['loss']):
        #     print epoch+1, val

        # print "\nRunning Test:"
        # res = model.predict_classes([X_test, X_test])
        # res_probs = model.predict_proba([X_test, X_test], verbose=0)
        # B-LSTM Model End -----------------------------------------------------

        id_to_word = {v: k for k, v in d.get_vocabulary().iteritems()}
        inv_y = {v: k for k, v in y_map.iteritems()}

        label_keys = ["PADDING", "A0", "A1", "A2", "A3", "NONE"]
        results = []
        for test_sent, test_labels, tets_preds, test_arg_bools, test_res_probs in zip(X_test, Y_test_raw, res, args_bools, res_probs):
            for word_id, label, pred, arg_bool, test_res_prob in zip(test_sent, test_labels, tets_preds, test_arg_bools, test_res_probs):
                if word_id > 0:
                    word_probs = dict(zip(label_keys, test_res_prob))
                    A0P = word_probs['A0']
                    A1P = word_probs['A1']
                    A2P = word_probs['A2']
                    A3P = word_probs['A3']
                    NONEP = word_probs['NONE'] + word_probs['PADDING']
                    SUMP = A0P + A1P + A2P + A3P + NONEP
                    if inv_y[pred] == "PADDING":
                        results.append([id_to_word[word_id],
                                        arg_bool, inv_y[label], "NONE",
                                        A0P/SUMP, A1P/SUMP, A2P/SUMP, A3P/SUMP,
                                        NONEP/SUMP])
                    else:
                        results.append([id_to_word[word_id],
                                        arg_bool, inv_y[label], inv_y[pred],
                                        A0P/SUMP, A1P/SUMP, A2P/SUMP, A3P/SUMP,
                                        NONEP/SUMP])

        raw_df = pd.DataFrame(test_map_data, columns=["SENT_ID", "TEXT", "GOLD"])
        res_df = pd.DataFrame(results, columns=["TEXT", "ARG", "GOLD", "PREDICTED", "A0", "A1", "A2", "A3", "NONE"])

        def correct(row):
            if (row['GOLD'] == row['PREDICTED']):
                return 1
            else:
                return 0

        res_df["CORRECT"] = res_df.apply(correct, axis=1)

        res_df = pd.concat([res_df, pd.DataFrame(raw_df.SENT_ID)], axis=1)
        res_df = res_df[["SENT_ID", "TEXT", "ARG", "GOLD", "PREDICTED", "CORRECT", "A0", "A1", "A2", "A3", "NONE"]]

        df = pd.read_csv("cleaned_input_data.tsv", sep="\t")

        def get_process(row):
            cur_sentid = row['SENT_ID']
            cur_df = df[df.SENT_ID == cur_sentid]
            for rid, rrow in cur_df.iterrows():
                    return rrow['PROCESS']
            return -1

        def get_argid(row):
            cur_span = row['TEXT']
            cur_sentid = row['SENT_ID']
            cur_df = df[df.SENT_ID == cur_sentid]
            for rid, rrow in cur_df.iterrows():
                if rrow['SPAN'] == cur_span:
                    return rrow['ARG_ID']
            return -1

        def get_sentence(row):
            cur_sentid = row['SENT_ID']
            cur_df = df[df.SENT_ID == cur_sentid]
            for rid, rrow in cur_df.iterrows():
                    return rrow['SENTENCE']
            return -1

        def get_start_idx(row):
            cur_span = row['TEXT']
            cur_sentid = row['SENT_ID']
            cur_df = df[df.SENT_ID == cur_sentid]
            for rid, rrow in cur_df.iterrows():
                if rrow['SPAN'] == cur_span:
                    return rrow['START_IDX']
            return -1

        def get_end_idx(row):
            cur_span = row['TEXT']
            cur_sentid = row['SENT_ID']
            cur_df = df[df.SENT_ID == cur_sentid]
            for rid, rrow in cur_df.iterrows():
                if rrow['SPAN'] == cur_span:
                    return rrow['END_IDX']
            return -1

        res_df["PROCESS"] = res_df.apply(get_process, axis=1)
        res_df["ARG_ID"] = res_df.apply(get_argid, axis=1)
        res_df["SENTENCE"] = res_df.apply(get_sentence, axis=1)
        res_df["START_IDX"] = res_df.apply(get_start_idx, axis=1)
        res_df["END_IDX"] = res_df.apply(get_end_idx, axis=1)
        res_df = res_df[["PROCESS", "SENT_ID", "SENTENCE", "ARG_ID", "START_IDX", "END_IDX", "TEXT", "ARG", "GOLD", "PREDICTED", "CORRECT", "A0", "A1", "A2", "A3", "NONE"]]

        # Dump RNN result if needed for analysis
        # res_df.to_csv("RNN_RESULT.tsv", sep="\t", index=False)

        # dump JSON file in appropriate format in order to run ILP
        j_dump_data = []
        g_process = res_df.groupby("PROCESS")
        processes = g_process.groups.keys()
        for process in processes:
            gi_process = g_process.get_group(process)
            g_sentence = gi_process.groupby("SENT_ID")
            sentence_ids = g_sentence.groups.keys()
            sent_list = []
            for sentence_id in sentence_ids:
                gi_sentence = g_sentence.get_group(sentence_id)
                # filter and only consider argument spans
                gif_sentence = gi_sentence[gi_sentence.ARG_ID > -1]
                if len(g_sentence.groups.keys()) > 0:
                    sentence_text = g_sentence.groups.keys()[0]

                    arg_ids = list(set(gif_sentence.ARG_ID.tolist()))
                    arg_list = []
                    for arg_id in arg_ids:
                        g_arg = gif_sentence[gif_sentence.ARG_ID == arg_id]
                        start_idx = g_arg["START_IDX"].tolist()[0]
                        end_idx = g_arg["END_IDX"].tolist()[0]
                        span = g_arg["TEXT"].tolist()[0]
                        gold = g_arg["GOLD"].tolist()[0]
                        predicted = g_arg["PREDICTED"].tolist()[0]
                        a0_p = g_arg["A0"].tolist()[0]
                        a1_p = g_arg["A1"].tolist()[0]
                        a2_p = g_arg["A2"].tolist()[0]
                        a3_p = g_arg["A3"].tolist()[0]
                        none_p = g_arg["NONE"].tolist()[0]
                        role_probs = [{"A0": a0_p},
                                      {"A1": a1_p},
                                      {"A2": a2_p},
                                      {"A3": a3_p},
                                      {"NONE": none_p}]
                        arg_list.append({'argId': arg_id,
                                         'text': span,
                                         'annotatedRole': gold,
                                         'annotatedLabel': 1,
                                         'rolePredicted': predicted,
                                         'startIdx': start_idx,
                                         'endIdx': end_idx,
                                         'probRoles': role_probs})
                    sent_list.append({'sentenceId': sentence_id,
                                      'text': sentence_text,
                                      'predictionArgumentSpan': arg_list})
            j_dump_data.append({'process': process,
                                'sentences': sent_list})

        out_name = 'out_data/fold-' + str(f_num+1) + '/test/'
        with open(out_name + 'test.srlpredict.json', 'w') as fp:
            json.dump(j_dump_data, fp, indent=4)
        with open(out_name + 'test.srlout.json', 'w') as fp:
            json.dump(j_dump_data, fp, indent=4)

    print "\n\nDone!"


if __name__ == '__main__':
    main()
