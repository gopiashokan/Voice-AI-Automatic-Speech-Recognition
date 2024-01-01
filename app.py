from transformers import pipeline

def get_transcription(filename: str):

    if not isinstance(filename, str):
        raise TypeError("Argument 'filename' must be a string.")
    
    transcription = pipeline("automatic-speech-recognition", model="gopiashokan/whisper-small-mr")(filename)

    return transcription
