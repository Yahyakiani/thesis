import nltk
import numpy as np
import sqlite3

LEMMATIZER = nltk.WordNetLemmatizer()

class WordPreprocessor:
    def __init__(self):
        self.db_conn = sqlite3.connect('word_information.db')


    def get_feature_vector(self, word):
        """
        Returns a numpy array of features given a word. This array of features,
        combined with a label of whether the word is seen as "complex," is given
        to a machine learning model as an example. The trained model will then
        be able to give predict whether a word is "complex."
        """
        word = word.lower()
        lemma = LEMMATIZER.lemmatize(word)
        return np.array([[
            len(lemma),
            self._get_age_of_acquisition(word),
            self._get_concreteness_score(word),
            self._get_unilem_count(lemma),
            self._get_unigram_count(word),
        ]])


    def _get_age_of_acquisition(self, word):
        """
        Returns an average age-of-acquisition for a given word.

        Database source: V Kuperman et al. 2012. Age-of-acquisition ratings for
        30,000 english words. Behavior Research Methods, 44(4):978–990.
        """
        c = self.db_conn.cursor()
        c.execute("SELECT rating_mean FROM age_of_acquisition WHERE word = ?",
            (word,))
        result = c.fetchone()
        return float(result[0]) if result else 0


    def _get_concreteness_score(self, word):
        """
        Returns a concreteness score, from 1 to 5, for a given word. 1
        represents "not concrete," and 5 represents "very concrete."

        Database source:  M Brysbaert et al. 2013. Concreteness ratings for 40
        thousand generally known english word lemmas. Behavior Research Methods,
        46(3):904–911.
        """
        c = self.db_conn.cursor()
        c.execute("SELECT conc_mean FROM concreteness WHERE word = ?",
            (word,))
        result = c.fetchone()
        return float(result[0]) if result else 0


    def _get_unilem_count(self, lemma):
        """
        Returns a frequency count for the given lemma in a corpus.

        Corpus: M Davies. 2008. The corpus of contemporary
        american english: 520 million words, 1990-present.
        """
        c = self.db_conn.cursor()
        c.execute("SELECT frequency FROM unigram_count WHERE lemma = ?",
            (lemma,))
        result = c.fetchone()
        return float(result[0]) if result else 0


    def _get_unigram_count(self, word):
        """
        Returns a frequency count for the given word in a corpus.

        Corpus: M Davies. 2008. The corpus of contemporary
        american english: 520 million words, 1990-present.
        """
        c = self.db_conn.cursor()
        c.execute("SELECT frequency FROM unigram_count WHERE word = ?",
            (word,))
        result = c.fetchone()
        return float(result[0]) if result else 0
