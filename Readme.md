# Reading Help and Assessment

## Project Goals

This project aims to enhance reading education by leveraging the power of Natural Language Processing (NLP) and Machine Learning (ML). Our objectives include:

### Personalized Reading Support

- Develop an NLP-powered system that adapts to individual reading levels and subjects.
- Provide real-time assistance for comprehension and vocabulary enhancement.

### Innovative Assessment Methods

- Utilize ML models and NLP techniques to assess reading proficiency beyond traditional testing methods.
- Design unified evaluation metrics for comprehensive assessment.

### Inclusive Technology Integration

- Create engaging reading platforms tailored to each child's unique learning pace and style.
- Support diverse learning needs with inclusive digital tools.
- Focus on overcoming accessibility challenges for seamless educational integration.

## Current Focus: Assessing Expressive Oral Reading Fluency

### Overview

Our current phase focuses on analyzing children's expressive oral reading fluency through audio recordings. This involves assessing reading accuracy, prosody, and other aspects crucial for reading fluency, aiming to provide immediate, actionable feedback.

### Technical Approach

The project employs a Python-based, multi-step approach integrating `pydub`, `whisper`, `librosa`, and `parselmouth` libraries to process and analyze reading performances:

1. **Audio Processing**: Convert MP3 recordings into WAV format for analysis.
2. **Transcription**: Use the `whisper` model for audio-to-text transcription to assess reading accuracy.
3. **Prosodic Feature Analysis**: Analyze pitch, intensity, duration, and speech rate using `librosa` and `parselmouth`.
4. **Feedback Generation**: Generate feedback combining accuracy and prosodic analysis to guide reading fluency improvements.

### Achievements in Project Goals

- **Personalized Support**: Offers tailored feedback based on individual reading performances.
- **Innovative Assessment**: Evaluates expressive oral reading fluency, enhancing traditional reading proficiency assessments.
- **Inclusive Technology**: Utilizes accessible AI models and technologies to cater to diverse learning needs.

