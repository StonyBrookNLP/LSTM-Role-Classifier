import json
import pandas as pd
from os.path import join


def load_srl_data(filename):
    d = json.load(open(filename, "r"))
    data = []
    for p_data in d:
        process = p_data['process']
        ss_data = p_data['sentences']
        process_list = []
        for s_data in ss_data:
            s_id = s_data['sentenceId']
            sentence = s_data['text']
            a_spans = s_data['annotatedArgumentSpan']
            for a_span in a_spans:
                annotation = a_span['annotatedRole']
                start_idx = a_span['startIdx']
                end_idx = a_span['endIdx']
                arg_text = a_span['text']
                arg_id = a_span['argId']
                process_list.append([process, s_id, sentence, arg_id, start_idx, end_idx, arg_text, annotation])
        data.extend(process_list)
    return data


def main():
    all_data = []

    for f, fold_dir in enumerate(["fold-1", "fold-2", "fold-3", "fold-4", "fold-5"]):
        fold_path = join("dataset", fold_dir)
        d_gold_file = join(fold_path, 'test', 'test.srlout.json')
        d_gold = load_srl_data(d_gold_file)
        all_data.extend(d_gold)

    df = pd.DataFrame(all_data, columns=['PROCESS', 'SENT_ID', 'SENTENCE', 'ARG_ID', 'START_IDX', 'END_IDX', 'SPAN', 'ROLE'])
    # This outputs dataset which needs to be processed to remove overlapping
    # argument spans
    df.to_csv("dirty_data.tsv", sep="\t")

if __name__ == '__main__':
    main()
