# Very hacky script to generate argument similarity cache file for ILP
import cPickle as pickle
import json
import numpy as np
import os

from scipy import linalg


def get_similarity(a, b):
    return np.dot(a, b)/linalg.norm(a)/linalg.norm(b)


embedding_map = pickle.load(open("embedding_map", "rb"))
EMBEDDING_DIM = 100

for fold in os.listdir("out_data1/"):
    file_name = "out_data1/" + fold + "/test/test.srlout.json"
    sim_file_name = fold + "lstm_sim.p"
    j_dump_data = json.load(open(file_name, "r"))
    sim_data = {}
    for process_data in j_dump_data:
        sentences = process_data['sentences']
        for sentence_data1 in sentences:
            sentence_spans1 = sentence_data1['predictionArgumentSpan']
            for sentence_span1 in sentence_spans1:
                span1 = sentence_span1['text']
                for sentence_data2 in sentences:
                    sentence_spans2 = sentence_data2['predictionArgumentSpan']
                    for sentence_span2 in sentence_spans2:
                        span2 = sentence_span2['text']
                        if span1 in embedding_map:
                            v_span1 = embedding_map[span1]
                        else:
                            v_span1 = np.zeros((EMBEDDING_DIM))
                        if span2 in embedding_map:
                            v_span2 = embedding_map[span2]
                        else:
                            v_span2 = np.zeros((EMBEDDING_DIM))
                        sim = get_similarity(v_span1, v_span2)
                        sim_data[(span1, span2)] = sim
                        sim_data[(span2, span1)] = sim
    pickle.dump(sim_data, open(sim_file_name, "wb"))


f1_sim = pickle.load(open("fold-1lstm_sim.p", "rb"))
f2_sim = pickle.load(open("fold-2lstm_sim.p", "rb"))
f3_sim = pickle.load(open("fold-3lstm_sim.p", "rb"))
f4_sim = pickle.load(open("fold-4lstm_sim.p", "rb"))
f5_sim = pickle.load(open("fold-5lstm_sim.p", "rb"))

f1_sim.update(f2_sim)
f1_sim.update(f3_sim)
f1_sim.update(f4_sim)
f1_sim.update(f5_sim)

pickle.dump(f1_sim, open("lstm_sim.p", "wb"))
