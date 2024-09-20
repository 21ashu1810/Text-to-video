
import librosa
import numpy as np
from moviepy.editor import VideoFileClip, ImageSequenceClip, CompositeVideoClip, AudioFileClip
from PIL import Image, ImageDraw
import os
from gtts import gTTS
from googletrans import Translator


def extract_text_from_pdf(pdf_path):
    import fitz  
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def translate_text(text, target_language):
    translator = Translator()
    translated = translator.translate(text, dest=target_language)
    return translated.text

def save_text_to_mp3(text, output_audio_path, lang='en'):
    tts = gTTS(text, lang=lang)
    tts.save(output_audio_path)

pdf_path = "/content/sodapdf-converted (2) (1).pdf"  
audio_path = "/content/output_audio.mp3"
character_video_path = "/content/extended_character (7).mp4"  
output_video_path = "/content/lip_synced_output.mp4"
target_language = 'hi'  

extracted_text = extract_text_from_pdf(pdf_path)
translated_text = translate_text(extracted_text, target_language)

save_text_to_mp3(translated_text, audio_path, lang=target_language)
print(f"Translated text: {translated_text[:500]}...")  
print(f"Audio saved to {audio_path}")

#Analyze Audio with Librosa
y, sr = librosa.load(audio_path, sr=None)

# Use a more detailed analysis by calculating the short-time Fourier transform (STFT)
stft = np.abs(librosa.stft(y))

# Calculate the spectral centroid 
centroids = librosa.feature.spectral_centroid(S=stft, sr=sr)[0]

# Normalize centroid values to match mouth shape range
centroids_normalized = np.interp(centroids, (centroids.min(), centroids.max()), (0, 1))

# Create Mouth Shapes Based on Centroids
def create_mouth_image(open_level):
    img = Image.new('RGB', (100, 100), color='white')
    draw = ImageDraw.Draw(img)
    mouth_height = int(10 + 20 * open_level)
    draw.rectangle([30, 100 - mouth_height - 40, 70, 100 - 40], fill='black')  # Mouth shape
    return img

# Create the mouth images sequence
mouth_images = [create_mouth_image(level) for level in centroids_normalized]

# Combine with Video
# Load the character video
character_clip = VideoFileClip(character_video_path).subclip(0, len(y) / sr)

# Convert PIL images to numpy arrays and create a video sequence
fps = sr / len(mouth_images)  # Frames per second should match the audio sample rate
mouth_sequence = ImageSequenceClip([np.array(img) for img in mouth_images], fps=fps)

# Adjust mouth_sequence duration to match character_clip
mouth_sequence = mouth_sequence.set_duration(character_clip.duration)

# Overlay mouth sequence onto character video
final_clip = CompositeVideoClip([character_clip, mouth_sequence.set_position(("center", "bottom"))])

# Add the original audio
final_clip = final_clip.set_audio(AudioFileClip(audio_path))

# Write the final video
final_clip.write_videofile(output_video_path, codec="libx264", audio_codec="aac")

# View the Generated Lip-Synced Video
from IPython.display import Video, display
if os.path.exists(output_video_path):
    display(Video(output_video_path, embed=True))
else:
    print("Cannot display the video as the file does not exist.")
