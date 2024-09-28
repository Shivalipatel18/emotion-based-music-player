import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import numpy as np
import mediapipe as mp
import os
import random
import pygame
import subprocess
from keras.models import load_model

# Initialize Pygame mixer
pygame.mixer.init()

# Define the emotions
emotions = ['neutral', 'fearful', 'happy', 'sad', 'surprise']

# Load your music files based on emotions
music_directory = 'C:/musicproject'  # Update with your music directory


# Load the emotion detection model and labels
model = load_model("model.h5")
label = np.load("labels.npy")

# Initialize Mediapipe holistic and hands models
holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils

# Initialize the EmotionProcessor class
class EmotionProcessor:
    def __init__(self, music_directory):
        self.music_directory = music_directory
        self.current_playlist = None
        self.current_index = 0

    def load_random_music(self, emotion, index=None):
        if emotion in music_files:
           self.current_playlist = music_files[emotion]
           if index is not None:
                self.current_index = index
           else:
                self.current_index = (self.current_index + 1) % len(self.current_playlist)
           music_path = os.path.join(self.music_directory, self.current_playlist[self.current_index])
           pygame.mixer.music.load(music_path)
           pygame.mixer.music.play()
        else:
            print("No playlist found for the detected emotion.")

    def play_next_song(self):
        if self.current_playlist:
            self.current_index = (self.current_index + 1) % len(self.current_playlist)
            music_path = os.path.join(self.music_directory, self.current_playlist[self.current_index])
            pygame.mixer.music.load(music_path)
            pygame.mixer.music.play()

# Initialize the EmotionProcessor instance
emotion_processor = EmotionProcessor(music_directory)

# Streamlit app setup
st.header("Emotion Based Music Player")
st.markdown("Welcome to My Music Player")
if "run" not in st.session_state:
    st.session_state["run"] = True

try:
    emotion = np.load("emotion.npy")[0]
except:
    emotion = ""

class VideoProcessor:
    def recv(self, frame):
        frm = frame.to_ndarray(format="bgr24")
        frm = cv2.flip(frm, 1)
        res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
        # Emotion detection code here...
        lst = []

        if res.face_landmarks:
            for i in res.face_landmarks.landmark:
                lst.append(i.x - res.face_landmarks.landmark[1].x)
                lst.append(i.y - res.face_landmarks.landmark[1].y)

            if res.left_hand_landmarks:
                for i in res.left_hand_landmarks.landmark:
                    lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
            else:
                for i in range(42):
                    lst.append(0.0)

            if res.right_hand_landmarks:
                for i in res.right_hand_landmarks.landmark:
                    lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
            else:
                for i in range(42):
                    lst.append(0.0)

            lst = np.array(lst).reshape(1, -1)

            pred = label[np.argmax(model.predict(lst))]

            cv2.putText(frm, pred, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            np.save("emotion.npy", np.array([pred]))

            drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_TESSELATION,
                                   landmark_drawing_spec=drawing.DrawingSpec(color=(0, 255, 0), thickness=-1,
                                                                             circle_radius=1),
                                   connection_drawing_spec=drawing.DrawingSpec(color=(0, 255, 0), thickness=1))
            drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
            drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

        return av.VideoFrame.from_ndarray(frm, format="bgr24")

# Emotion detection and music playback logic
if st.session_state["run"]:
    webrtc_streamer(key="emotion", desired_playing_state=True, video_processor_factory=VideoProcessor)

btn_play = st.button("Play songs")
pl = "C:/Program Files (x86)/Windows Media Player/wmplayer.exe"
if btn_play:
    if not emotion:
        st.warning("Please let me capture your emotion first")
        st.session_state["run"] = True
    else:
        if emotion == 'happy':
            randomfile = random.choice(os.listdir("C:/musicproject/happy/"))
            st.popover('You are happy! I will play a special song: ' + randomfile)
            file = ('C:/musicproject/happy/' + randomfile)
            subprocess.call([pl, file])
        if emotion == 'sad':
            randomfile = random.choice(os.listdir("C:/musicproject/sad/"))
            st.popover('You are sad!:( I will play a song: ' + randomfile)
            file = ('C:/musicproject/sad/' + randomfile)
            subprocess.call([pl, file])
        if emotion == 'neutral':
            randomfile = random.choice(os.listdir("C:/musicproject/neutral/"))
            st.popover('You are Neutral! lets Enjoy the song: ' + randomfile)
            file = ('C:/musicproject/neutral/' + randomfile)
            subprocess.call([pl, file])   
        if emotion == 'fearful':
            randomfile = random.choice(os.listdir("C:/musicproject/fearful/"))
            st.popover('You are fearful! Now time for more scary songs: ' + randomfile)
            file = ('C:/musicproject/fearful/' + randomfile)
            subprocess.call([pl, file])
        if emotion == 'angry':
            randomfile = random.choice(os.listdir("C:/musicproject/angry/"))
            st.popover('You are angry! have some energetic songs: ' + randomfile)
            file = ('C:/musicproject/angry/' + randomfile)
            subprocess.call([pl, file]) 
#btn_stop = st.button("Stop songs")
#if btn_stop:
    #pygame.mixer.music.stop()

#btn_pause = st.button("Pause")
#if btn_pause:
   # pygame.mixer.music.pause()

#btn_resume = st.button("Resume")
#if btn_resume:
    #pygame.mixer.music.unpause()

#btn_next = st.button("Next")
#if btn_next:
    #emotion_processor.play_next_song()
