import nltk
import sys
from glob import glob
from os import path
from string import punctuation
from itertools import chain
from math import log


FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    files = {}
    for filename in glob(path.join(directory, "*.txt")):
        with open(filename, "r", encoding="utf8") as f:
            fname = path.split(f.name)[1]
            files[fname] = f.read()

    return files


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    tokens = nltk.word_tokenize(document.lower())
    stop_words = set(nltk.corpus.stopwords.words('english'))
    words = [tokes for tokes in tokens if tokes not in punctuation and tokes not in stop_words]
    return words


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    word_dict = {}
    words_list = list(documents.values())
    words = list(chain.from_iterable(words_list))
    no_of_docu = len(documents)
    for word in words:
        count = 0
        for w_list in words_list:
            if word in w_list:
                count += 1

        word_dict[word] = log(no_of_docu / count)

    return word_dict


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    file_dict = {}
    for file in files:
        tf_idf = 0
        for q in query:
            if q in files[file]:
                freq = files[file].count(q)
                tf_idf += freq * idfs[q]

        file_dict[file] = tf_idf

    sorted_tfidf = sorted(file_dict, key=file_dict.__getitem__, reverse=True)
    return sorted_tfidf[:n]


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    sen_dict = {}
    for sen, words in sentences.items():
        words_in_query = query.intersection(words)
        match_word = 0
        for q in words_in_query:
            match_word += idfs[q]
        sen_dict[sen] = {'idf': match_word, 'den': len(words_in_query) / len(words)}

    sorted_idf = sorted(sen_dict.items(), key=lambda x: (x[1]['idf'], x[1]['den']), reverse=True)
    sorted_idf = [val[0] for val in sorted_idf]
    return sorted_idf[:n]


if __name__ == "__main__":
    main()
