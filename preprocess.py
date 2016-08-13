import os
import numpy as np
import pandas as pd

from nltk import word_tokenize
from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')


class Embeddings(object):
    def __init__(self, GLOVE_DIR=""):
        self.unavailable = []
        self.available = []
        self.GLOVE_DIR = GLOVE_DIR
        self.embeddings_index = {}
        f = open(os.path.join(GLOVE_DIR, 'glove_vectors.txt'))
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            self.embeddings_index[word] = coefs
        f.close()

    def get_embedding(self, input_obj):
        # if a string is passed, do the following
        if isinstance(input_obj, str):
            if input_obj in self.embeddings_index:
                return self.embeddings_index[input_obj]
            else:
                fine_tokens = tokenizer.tokenize(input_obj)
                obj_vectors = []
                for word in fine_tokens:
                    if word in self.embeddings_index:
                        obj_vectors.append(self.embeddings_index[word])
                        self.available.append(word)
                    else:
                        self.unavailable.append(word)
                if obj_vectors:
                    return reduce(lambda x, y: x*y, obj_vectors)
                else:
                    # return APPROPRIATE RANDOM/SOME EMBEDDING
                    return None

        # if a list of strings is passed, do the following
        elif isinstance(input_obj, list):
            obj_vectors = []
            for word in input_obj:
                if word in self.embeddings_index:
                    obj_vectors.append(self.embeddings_index[word])
                    self.available.append(word)
                else:
                    fine_tokens = tokenizer.tokenize(word)
                    for word2 in fine_tokens:
                        if word2 in self.embeddings_index:
                            obj_vectors.append(self.embeddings_index[word2])
                            self.available.append(word)
                        else:
                            self.unavailable.append(word2)
            if obj_vectors:
                return reduce(lambda x, y: x*y, obj_vectors)
            else:
                # TODO: return APPROPRIATE RANDOM/SOME EMBEDDING
                return None

    def get_unknown(self):
        return self.unavailable

    def get_known(self):
        return self.available


class WordToken(object):
    def __init__(self, token, token_id, embedding, label, arg):
        self.token = token
        self.token_id = token_id
        self.label = label
        self.embedding = embedding
        self.arg = arg

    def __repr__(self):
        return "WordToken: (%d)(%s)(%s)" % (self.get_token_id(),
                                            self.get_token(),
                                            self.get_label())

    def get_token(self):
        return self.token

    def get_token_id(self):
        return self.token_id

    def get_label(self):
        return self.label

    def get_embedding(self):
        return self.embedding


class Sentence(object):
    def __init__(self, sentence_id, text, tokens, token_ids, token_embeddings, labels, args):
        self.sentence_id = sentence_id
        self.text = text
        self.tokens = []
        for token, token_id, embedding, label, arg in zip(tokens, token_ids, token_embeddings, labels, args):
            self.tokens.append(WordToken(token, token_id, embedding, label, arg))

    def __iter__(self):
        return iter(self.tokens)

    def __len__(self):
        return len(self.tokens)

    def __repr__(self):
        return "Sentence: (%s) (%s)" % (self.sentence_id, self.text)

    def get_sentence(self):
        return self.text

    def get_tokens(self):
        return self.tokens


class Process(object):
    def __init__(self, process):
        if not isinstance(process, str):
            raise TypeError, "process name must be of string type"
        self.name = process
        self.sentences = []

    def __iter__(self):
        return iter(self.sentences)

    def __len__(self):
        return len(self.sentences)

    def __repr__(self):
        return "Process: (%s) has (%d) sentences" % (self.name, len(self.sentences))

    def add_sentence(self, sentence):
        if not isinstance(sentence, Sentence):
            raise TypeError, "sentence must be a Sentence instance"
        self.sentences.append(sentence)

    def get_sentences(self):
        return self.sentences


class Dataset(object):
    def __init__(self, filename, sep="\t"):
        self.vocabulary = {}
        self.vocabulary_size = 1
        self.processes = []
        self.objEmbeddings = Embeddings()
        self.load_dataset(filename, sep)

    def __iter__(self):
        return iter(self.processes)

    def __repr__(self):
        return "Dataset object contains (%d) processes" % (len(self.processes))

    def load_dataset(self, filename, sep):
        df = pd.read_csv(filename, sep=sep)

        process_gdf = df.groupby('PROCESS')
        processes = process_gdf.groups.keys()

        for p_id, process in enumerate(processes):
            # create process object
            objProcess = Process(process)
            process_df = process_gdf.get_group(process)
            sent_gdf = process_df.groupby('SENT_ID')
            sent_ids = sent_gdf.groups.keys()
            for sent_id in sent_ids:
                sent_df = sent_gdf.get_group(sent_id)
                # initialize last idx to 0 (start of the sentence)
                last_idx = 0
                # create a tokens and labels by considering phrases and annotations
                tokens = []
                token_ids = []
                token_embeddings = []
                labels = []
                args = []
                for r_id, row in sent_df.iterrows():
                    # extract data from each row
                    process = row['PROCESS'].lower()
                    sent_id = row['SENT_ID']
                    sentence = row['SENTENCE'].lower()
                    start_idx = row['START_IDX']
                    end_idx = row['END_IDX']
                    span = row['SPAN'].lower()
                    role = row['ROLE']
                    # tokenize the sentence
                    t = word_tokenize(sentence)
                    # add spans from last index till the current span
                    prev_spans = t[last_idx:start_idx-1]
                    for p_token in prev_spans:
                        tokens.append(p_token)
                        args.append(False)
                        token_embeddings.append(self.objEmbeddings.get_embedding(p_token))
                        labels.append('NONE')
                        if p_token in self.vocabulary:
                            token_ids.append(self.vocabulary[p_token])
                        else:
                            self.vocabulary[p_token] = self.vocabulary_size
                            self.vocabulary_size += 1
                            token_ids.append(self.vocabulary[p_token])
                    # add the current span
                    tokens.append(span)
                    args.append(True)
                    token_embeddings.append(self.objEmbeddings.get_embedding(t[start_idx-1:end_idx]))
                    labels.append(role)
                    if span in self.vocabulary:
                        token_ids.append(self.vocabulary[span])
                    else:
                        self.vocabulary[span] = self.vocabulary_size
                        self.vocabulary_size += 1
                        token_ids.append(self.vocabulary[span])
                    # update last idx
                    last_idx = end_idx
                # rest of the sentence after the last annotated span's end index
                if last_idx < len(t):
                    remaining_spans = t[last_idx:]
                    for r_token in remaining_spans:
                        tokens.append(r_token)
                        args.append(False)
                        token_embeddings.append(self.objEmbeddings.get_embedding(r_token))
                        labels.append('NONE')
                        if r_token in self.vocabulary:
                            token_ids.append(self.vocabulary[r_token])
                        else:
                            self.vocabulary[r_token] = self.vocabulary_size
                            self.vocabulary_size += 1
                            token_ids.append(self.vocabulary[r_token])
                # create sentence object and add it to process object
                objSentence = Sentence(sent_id, sentence, tokens, token_ids, token_embeddings, labels, args)
                objProcess.add_sentence(objSentence)
            self.processes.append(objProcess)

    def get_vocabulary(self):
        return self.vocabulary
