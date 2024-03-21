from predictor import WordComplexityPredictor

# Initialize the predictor.
predictor = WordComplexityPredictor(debug=True)

# Path to your MP3 file.
mp3_file_path = "audio\output_003.wav"
# original_text = "once last year i found an armour and a dog"

# Process the audio file and get word complexities.
word_complexities = predictor.process_audio_and_predict(mp3_file_path)

print("\nComplexities for words in the audio transcription (excluding common words):")
for word, complexity in word_complexities.items():
    print(f"{word:<15}\t\t{complexity}")
