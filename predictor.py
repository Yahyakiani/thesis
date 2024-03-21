import whisper
import pickle
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from pydub import AudioSegment
from preprocessor import WordPreprocessor

MODEL_FILENAME = "model.pickle"
TRAINING_CSV_FILENAME = "data/training_data.csv"
TREE_MAX_DEPTH = 4


class WordComplexityPredictor:
    def __init__(self, debug=False):
        self._preprocessor = WordPreprocessor()
        self._debug = debug
        self.common_words = set(["a", "the", "of"])

        if not self._load_model():
            # File not found, just create a new model and train it.
            self._model = DecisionTreeRegressor(max_depth=TREE_MAX_DEPTH)
            self._train_and_save(TRAINING_CSV_FILENAME)

        self._log("Constructed analyzer.")

    def _load_model(self):
        """
        Loads a scikit-learn machine learning model from a Pickle file. Returns
        True if the model was successfully loaded.
        """
        try:
            f = open(MODEL_FILENAME, "rb")
            self._model = pickle.load(f)
            self._log("Loaded model from file.")
            return True
        except FileNotFoundError as e:
            return False

    def _train_and_save(self, train_csv):
        """
        Trains a scikit-learn machine learning model using the data provided in
        the filename given (train_csv) and preprocessing implemented in
        WordPreprocessor. After training, it saves the trained model to
        MODEL_FILENAME.

        train_csv is a CSV file with a header row. The column names are "word",
        "sentence" (that the word appears in), "index" (of the word in the
        sentence), and "label" (0 to 1, inclusive, that measures how complex
        that word is seen to be).
        """
        training_data = pd.read_csv(train_csv)
        X = np.ma.row_stack(
            [
                self._preprocessor.get_feature_vector(row["word"])
                for _, row in training_data.iterrows()
            ]
        )
        y = np.array([row["label"] for _, row in training_data.iterrows()])
        self._model.fit(X, y)

        f = open(MODEL_FILENAME, "wb")
        pickle.dump(self._model, f)
        self._log("Trained and saved model to file.")

    def predict(self, word):
        """
        Uses self.model to predict, on a scale of 0 to 1 inclusive, how complex
        is word is seen to be.
        """
        features = self._preprocessor.get_feature_vector(word)
        pred = self._model.predict(features)[0]
        self._log('Predicted "{}" with complexity {}.'.format(word, pred))
        return float(pred)

    def _log(self, message):
        if self._debug:
            print(message)

    # New Method: Convert MP3 to WAV
    def convert_mp3_to_wav(self, mp3_file_path, wav_file_path):
        audio = AudioSegment.from_mp3(mp3_file_path)
        audio.export(wav_file_path, format="wav")
        self._log(f"Converted {mp3_file_path} to {wav_file_path}.")

    # New Method: Transcribe Audio to Text
    def transcribe_audio(self, audio_file_path):
        model = whisper.load_model("base")
        result = model.transcribe(audio_file_path)
        self._log(f"Transcribed audio to text: {result['text']}")
        return result["text"]

    # New Method: Process Audio File and Predict Word Complexities
    def process_audio_and_predict(self, mp3_file_path):
        wav_file_path = mp3_file_path.replace(".mp3", ".wav")
        self.convert_mp3_to_wav(mp3_file_path, wav_file_path)
        transcription = self.transcribe_audio(wav_file_path)
        words = transcription.split()

        complexities = {}
        for word in words:
            if word.lower() not in self.common_words:
                complexity = self.predict(word)
                if complexity is not None:
                    complexities[word] = complexity

        return complexities
