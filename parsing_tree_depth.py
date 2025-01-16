from collections import Counter, deque

import pandas as pd
import spacy
from nltk import Tree
import benepar

def nl_con_parsing():
    # nl_df = pd.read_csv('./ud_data.csv')
    id_df = pd.read_csv('./evaluation_all.csv')
    # poses = df['pos'].values.tolist()
    # counter = Counter(poses)
    # print(counter.popitem())

    # import benepar, spacy
    nlp = spacy.load('en_core_web_sm')
    nlp.add_pipe('benepar', config={'model': 'benepar_en3'})
    # nlp1 = spacy.load('en_core_web_sm')
    # nlp1.add_pipe('benepar', config={'model': 'benepar_en3'})
    doc = nlp('get start time')
    doc1 = nlp('I congratulate School of Open teams across Africa for the innovative and transformative mode of teaching and learning that we are launching today .')
    sent = list(doc.sents)[0]
    sent1 = list(doc1.sents)[0]
    print(sent._.parse_string)
    print(sent1._.parse_string)
    cal_tree_depth(sent1._.parse_string)

    ids = id_df['SEQUENCE'].values.tolist()
    # nls = nl_df['sentences'].values.tolist()

    id_lens = 0
    nl_lens = 0

    for id in ids:
        doc = nlp(id)
        sent = list(doc.sents)[0]
        max_len = cal_tree_depth(sent._.parse_string)
        id_lens += max_len

    # for id in nls:
    #     doc = nlp(id)
    #     sent = list(doc.sents)[0]
    #     max_len = cal_tree_depth(sent._.parse_string)
    #     nl_lens += max_len

    print(id_lens/4862.0)
    # print(nl_lens/3000.0)



def cal_tree_depth(tree):
    stack = deque()
    depth = 0
    max_depth = depth
    for t in tree:
        if t == '(':
            stack.append(t)
            depth += 1
            if depth > max_depth:
                max_depth = depth
        elif t == ')':
            stack.pop()
            depth -= 1
    return max_depth


def check_phrases_in_tree(parse_string):
    """
    Recursively checks the tree to find any non-VP, NP, or PP phrases.
    """
    tree = Tree.fromstring(parse_string)
    non_vp_np_pp_phrases = []

    # Traverse the tree nodes
    for subtree in tree.subtrees():
        # If the subtree is not VP, NP, or PP, add it to the list
        if subtree.label() not in {"VP", "NP", "PP", "ROOT"}:
            non_vp_np_pp_phrases.append(subtree.label())

    return non_vp_np_pp_phrases

def parse_and_check_csv(input_csv, output_csv):
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe('benepar', config={'model': 'benepar_en3'})

    # Read CSV file
    df = pd.read_csv(input_csv)

    # Prepare an empty list to hold rows that contain phrases other than VP, NP, or PP
    rows_with_other_phrases = []

    # Process each row in the CSV
    for idx, row in df.iterrows():
        sequence = row['SEQUENCE']

        # Parse the sequence using SpaCy + Benepar
        doc = nlp(sequence)

        for sent in doc.sents:
            tree = sent._.parse_string  # Get the parse tree from Benepar
            print(sent, tree)

            # Check for non-VP, NP, PP phrases
            other_phrases = check_phrases_in_tree(tree)

            if other_phrases:  # If there are other phrases, save the row and the phrase structures
                rows_with_other_phrases.append({
                    'SEQUENCE': sequence,
                    'Other_Phrases': ", ".join(set(other_phrases))
                })

    # Convert the result to a DataFrame and save it to a new CSV file
    result_df = pd.DataFrame(rows_with_other_phrases)
    # result_df.to_csv(output_csv, index=False)


if __name__ == '__main__':
    # nl_con_parsing()
    parse_and_check_csv('./evaluation_all.csv', 'phrase_structure.csv')
