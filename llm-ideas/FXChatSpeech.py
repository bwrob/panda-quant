import speech_recognition as sr # recognise speech
import playsound # to play an audio file
from gtts import gTTS # google text to speech
import random
from time import ctime # get time details
import time
import os # to remove created audio files
from colorama import Fore, Back, Style

r = sr.Recognizer() # initialise a recogniser
# listen for audio and convert it to text:
def record_audio():
    with sr.Microphone() as source: # microphone as source
        r.adjust_for_ambient_noise(source)
        
        print(Style.BRIGHT + 'Ready and Listening')
        print(Style.RESET_ALL)
 
        audio = r.listen(source, phrase_time_limit=10)  # listen for the audio via source
        voice_data = ''
        try:
            voice_data = r.recognize_google(audio)  # convert audio to text
        except sr.UnknownValueError: # error: recognizer does not understand
            voice_data = ''
        except sr.RequestError:
            speak('Sorry, the service is down') # error: recognizer is not connected
            
        print(f">> {voice_data.lower()}") # print what user said
        return voice_data.lower()

# get string and make a audio file to be played
def speak(audio_string):
    tts = gTTS(text=audio_string, lang='en') # text to speech(voice)
    r = random.randint(1,20000000)
    audio_file = 'audio' + str(r) + '.mp3'
    tts.save(audio_file) # save as mp3
    playsound.playsound(audio_file) # play the audio file
    print(Style.BRIGHT + f"FloGPT : {audio_string}") # print what app said
    print(Style.RESET_ALL) 
    os.remove(audio_file) # remove audio file

from FXRisk import get_approval
from FXChatProcessor import chat_parser

def respond(voice_data):
    if voice_data == '':
        pass
    
    print(voice_data)
    res = chat_parser(voice_data)
    approval, risk, limit, cpty = get_approval(res)
    
    if approval:
        response = "your trade is approved with " + cpty
    else:
        response = "your trade is rejected with " + cpty

    speak(response)

time.sleep(1)

while(1):
    voice_data = record_audio() # get the voice input
    respond(voice_data) # respond
    break

