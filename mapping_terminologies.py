import argparse

import numpy as np
from nltk import PorterStemmer
from nltk.stem import SnowballStemmer, WordNetLemmatizer
import string
from nltk.corpus import stopwords
import pandas as pd

# Load stopwords for both languages
from string_grouper import match_most_similar, match_strings, compute_pairwise_similarities

french_stopwords = set(stopwords.words('french'))
english_stopwords = set(stopwords.words('english'))


def load_additional_stopwords(filepath):
    try:
        with open(filepath, "r", encoding='utf8') as f:
            return f.read().splitlines()
    except FileNotFoundError:
        print("Additional stopwords file not found.")
        return []


additionalFrenchStopWords = load_additional_stopwords("data/stopWords/additional_stop_words")
french_stopwords.update(additionalFrenchStopWords)


def remove_accents(input_str):
    trans = str.maketrans("éàèùâêîôûçæœ", "eaeuaeioucao")
    return input_str.translate(trans)


def preprocess_text(sentence, language='auto'):
    stopwords_set = french_stopwords if language == 'french' else english_stopwords
    stemmer = SnowballStemmer(language) if language == 'french' else PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    #print(sentence)
    words = sentence.lower().split()
    no_punc = [word.translate(str.maketrans('', '', string.punctuation)) for word in words]
    no_stops = [word for word in no_punc if word not in stopwords_set]
    stemmed = [stemmer.stem(remove_accents(word)) for word in no_stops]
    lemmatized = [lemmatizer.lemmatize(word) for word in stemmed]

    return " ".join(lemmatized)

def preprocess_dataframe(df, column_name, stemmer, lemmatizer, stopwords):
    """
    Preprocess all entries in a DataFrame column.
    """
    return df[column_name].apply(lambda x: preprocess_text(x, stemmer, lemmatizer, stopwords))

def run_lexical_mapping(input_termino1_path, input_termino2_path, output_path, language='auto', match_most_s=True,
                        match_s=True):
    try:
        df_termino1 = pd.read_csv(input_termino1_path, delimiter=",",na_values='')
        df_termino2 = pd.read_csv(input_termino2_path, delimiter=",",na_values='')
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    # Ensure 'code' and 'label' columns exist
    for df in [df_termino1, df_termino2]:
        if 'code' not in df.columns or 'label' not in df.columns:
            raise ValueError("Input CSV must contain 'code' and 'label' columns.")

    #df_termino1 = df_termino1.replace(np.nan, '', regex=True)
    df_termino1['proc_labels_t1'] = df_termino1['label'].apply(lambda x: preprocess_text(x, language))
    df_termino1 = df_termino1.replace(np.nan, '', regex=True)
    proc_label1_t = pd.Series(df_termino1['proc_labels_t1'].values) ##pathos
    labels_t1 = pd.Series(df_termino1['label'].values)
    code_t1 = pd.Series(df_termino1['code'].values)

    # Load and preprocess Termino target

    #df_termino2 = df_termino2.replace(np.nan, '', regex=True)
    df_termino2['proc_labels_t2'] = df_termino2['label'].apply(lambda x: preprocess_text(x, language))
    df_termino2 = df_termino2.replace(np.nan, '', regex=True)
    #print(pd.Series(df_termino2['proc_labels_t2'].values))
    proc_label2_t = pd.Series(df_termino2['proc_labels_t2'].values) ##cim10
    labels_t2  = pd.Series(df_termino2['label'].values)
    code_t2 = pd.Series(df_termino2['code'].values)


    master_id = pd.Series([i for i in range(len(labels_t2))])
    duplicates_id = pd.Series([i for i in range(len(labels_t1))])

    if match_most_s:
        alignement_matching_most_similar = match_most_similar(proc_label2_t,
                                                              proc_label1_t,master_id,duplicates_id,
                                     ngram_size=3, min_similarity=0.5)

        #print(alignement_matching_most_similar)

        alignement_matching_most_similar= alignement_matching_most_similar.replace(np.nan, '', regex=True)
        alignement_matching_most_similar.loc[alignement_matching_most_similar.most_similar_index == "", 'most_similar_master'] = ""


        labels_org_t2 = []
        id_t2 = []
        for ind in alignement_matching_most_similar['most_similar_index']:
            if ind != "" :
                labels_org_t2.append(labels_t2[int(ind)])
                id_t2.append(code_t2[int(ind)])
            else:
                labels_org_t2.append("")
                id_t2.append("")

        alignement_matching_most_similar["termino_1_codes"] = code_t1
        alignement_matching_most_similar["termino_1_labels"] = labels_t1
        alignement_matching_most_similar["termino_2_labels"] = labels_org_t2
        alignement_matching_most_similar["termino_2_codes"] = id_t2

        #print(alignement_matching_most_similar)

        cosine = compute_pairwise_similarities(proc_label1_t,
                                               alignement_matching_most_similar['most_similar_master'])

        alignement_matching_most_similar.drop('most_similar_master_id', inplace=True, axis=1)
        alignement_matching_most_similar.drop('most_similar_index', inplace=True, axis=1)


        alignement_matching_most_similar = alignement_matching_most_similar[['termino_1_codes',
                                           'termino_1_labels',
                                          'termino_2_codes',
                                            'termino_2_labels']]
        alignement_matching_most_similar['similarity'] = cosine

        alignement_matching_most_similar.to_excel(f'{output_path}_most_similar.xlsx')

    if match_s:
        alignement_matching_multiple_similar = match_strings(df_termino1['proc_labels_t1'],
                                                              df_termino2['proc_labels_t2'],
                                                         max_n_matches=20, min_similarity=0.5)

        labels_org_t1 = []
        id_t1 = []
        labels_org_t2 = []
        id_t2 = []

        for ind in alignement_matching_multiple_similar['left_index']:
            if ind != "":
                labels_org_t1.append(labels_t1[ind])
                id_t1.append(code_t1[ind])
            else:
                labels_org_t1.append("")
                id_t1.append("")



        for ind in alignement_matching_multiple_similar['right_index']:
            if ind != "":
                labels_org_t2.append(labels_t2[ind])
                id_t2.append(code_t2[ind])
            else:
                labels_org_t2.append("")
                id_t2.append("")


        alignement_matching_multiple_similar["termino_1_codes"] = id_t1
        alignement_matching_multiple_similar["termino_1_labels"] = labels_org_t1
        alignement_matching_multiple_similar["termino_2_labels"] = labels_org_t2
        alignement_matching_multiple_similar["termino_2_codes"] = id_t2

        cosine = compute_pairwise_similarities(alignement_matching_multiple_similar['left_proc_labels_t1'],
                                               alignement_matching_multiple_similar['right_proc_labels_t2'])

        alignement_matching_multiple_similar['similarity'] = cosine

        alignement_matching_multiple_similar = alignement_matching_multiple_similar[['termino_1_codes',
                                                                           'termino_1_labels',
                                                                             'termino_2_codes',
                                                                             'termino_2_labels',"similarity"]]

        alignement_matching_multiple_similar.to_excel(f'{output_path}_multiple_similar.xlsx')

    # Similarity matching and results post-processing as before

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run lexical mapping between two terminologies.")
    parser.add_argument("--input1", required=True, help="Input file path for terminology 1")
    parser.add_argument("--input2", required=True, help="Input file path for terminology 2")
    parser.add_argument("--output", required=True, help="Output file path for the mapping results")
    parser.add_argument("--language", default="auto", choices=["auto", "french", "english"],
                            help="Language for preprocessing ('auto', 'french', 'english'). Default is 'auto'.")
    #args = parser.parse_args()
    #run_lexical_mapping(args.input1, args.input2, args.output, args.language)

    #main('data/terminos_test/termino1.csv', 'data/terminos_test/termino2.csv',language="french")

    run_lexical_mapping('data_pathos/patho_annot.csv','data_pathos/data_icd10.csv',
                        'output_mapping',language="french")
