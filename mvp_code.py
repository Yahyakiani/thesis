from pydub import AudioSegment
import whisper
import librosa
import parselmouth
import numpy as np


# Step 1: Convert MP3 to WAV
def convert_mp3_to_wav(mp3_file_path, wav_file_path):
    audio = AudioSegment.from_mp3(mp3_file_path)
    audio.export(wav_file_path, format="wav")


# Step 2: Transcribe Audio to Text with Whisper
def transcribe_audio(audio_file_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_file_path)
    return result["text"]


# Step 3: Compare Transcribed Text to Original Text
def compare_texts(original_text, transcribed_text):
    original_words = set(original_text.lower().split())
    transcribed_words = set(transcribed_text.lower().split())
    matching_words = original_words.intersection(transcribed_words)
    accuracy = len(matching_words) / len(original_words) * 100
    return accuracy


def analyze_audio_prosodic_features(audio_file_path):
    # Load the audio file
    y, sr = librosa.load(audio_file_path)

    # Use Parselmouth to analyze pitch and intensity
    snd = parselmouth.Sound(audio_file_path)
    pitch = snd.to_pitch()
    intensity = snd.to_intensity()

    # Extract pitch values and intensity values
    pitch_values = pitch.selected_array["frequency"]
    intensity_values = intensity.values.T[0]

    # Filter out unvoiced (0 Hz) pitch values for mean pitch calculation
    voiced_pitch_values = pitch_values[pitch_values != 0]
    mean_pitch = np.mean(voiced_pitch_values) if len(voiced_pitch_values) > 0 else 0
    mean_intensity = np.mean(intensity_values)

    # Calculate the duration of the audio in seconds
    duration = librosa.get_duration(y=y, sr=sr)

    # Calculate the speech rate (number of voiced frames per second)
    speech_rate = len(voiced_pitch_values) / duration

    print(
        {
            "mean_pitch": mean_pitch,
            "mean_intensity": mean_intensity,
            "duration": duration,
            "speech_rate": speech_rate,
        }
    )

    return {
        "mean_pitch": mean_pitch,
        "mean_intensity": mean_intensity,
        "duration": duration,
        "speech_rate": speech_rate,
    }


def generate_feedback(accuracy, prosodic_features):
    feedback = ""

    # Feedback based on accuracy
    if accuracy > 90:
        feedback += "Excellent reading! "
    elif accuracy > 75:
        feedback += "Good job, but there's room for improvement. "
    else:
        feedback += "Keep practicing, and you'll improve. "

    # Feedback based on prosodic features
    # Assuming 'prosodic_features' is a dictionary with 'mean_pitch', 'mean_intensity', 'duration', and 'speech_rate'
    if (
        prosodic_features["speech_rate"] < 100
    ):  # Threshold values are examples and need to be adjusted
        feedback += "Try to read a bit faster. "
    elif prosodic_features["speech_rate"] > 160:
        feedback += "You're reading quite fast. Try to slow down a bit. "

    if (
        prosodic_features["mean_pitch"] > 250
    ):  # Example threshold, adjust based on your data
        feedback += "Your pitch is quite high, try to moderate it. "
    elif prosodic_features["mean_pitch"] < 100:
        feedback += (
            "Your pitch is low, try to enliven your reading by varying it more. "
        )

    # Example for intensity
    if (
        prosodic_features["mean_intensity"] < 45
    ):  # Example threshold, adjust based on your data
        feedback += "Your reading is a bit soft, try to speak louder. "
    elif prosodic_features["mean_intensity"] > 80:
        feedback += "Your reading is quite loud, try to speak a bit softer. "

    return feedback


def speech_feedback(age, speech_rate):
    """
    Provides feedback on speech rate based on the age of the speaker.
    This is a simplified model and should be adjusted with more comprehensive data for real-world applications.

    Args:
    age (int): The age of the speaker.
    speech_rate (int): The speech rate in words per minute (wpm).

    Returns:
    str: Feedback on the speaker's speech rate.
    """

    # Approximate age-related norms for speech rate (wpm)
    # These are simplified and should be refined with detailed research data
    age_norms = {
        (5, 8): (100, 140),  # Ages 5-8
        (9, 12): (140, 180),  # Ages 9-12
        (13, 17): (180, 220),  # Ages 13-17
    }

    # Find the appropriate age range
    for age_range, (min_rate, max_rate) in age_norms.items():
        if age_range[0] <= age <= age_range[1]:
            if speech_rate < min_rate:
                return f"Your speech rate is {speech_rate} wpm, which is slower than the average for your age group ({min_rate}-{max_rate} wpm). Try to speak a bit faster for better clarity."
            elif speech_rate > max_rate:
                return f"Your speech rate is {speech_rate} wpm, which is faster than the average for your age group ({min_rate}-{max_rate} wpm). Slowing down might help with clarity and understanding."
            else:
                return f"Your speech rate is {speech_rate} wpm, which is within the average range for your age group ({min_rate}-{max_rate} wpm). Keep up the good work!"

    return "Age out of range for this feedback system. Please enter an age between 5 and 17."


# Putting It All Together
def main(wav_file_path, original_text):

    # Transcribe the audio to text
    transcribed_text = transcribe_audio(wav_file_path)
    print("Transcribed Text:", transcribed_text)

    # Compare the transcribed text to the original text
    accuracy = compare_texts(original_text, transcribed_text)
    print(f"Reading Accuracy: {accuracy:.2f}%")

    prosodic_features = analyze_audio_prosodic_features(wav_file_path)
    feedback = generate_feedback(accuracy, prosodic_features)

    # Generate and print feedback
    print(feedback)


# Example Usage
if __name__ == "__main__":
    mp3_file_path = "audio\output_000.wav"
    original_text = (
        "The quick brown fox I got armor in a sword  jumps over the lazy dog."
    )
    main(mp3_file_path, original_text)
