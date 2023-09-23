# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 16:40:27 2020

@author: Madhavi
"""
import requests

URL="http://127.0.0.1:5000/predict"

test_audio_file_path='dataset/metal.00000.wav'  

if __name__=='__main__':
    audio_file=open(test_audio_file_path,"rb")
    
    values={"file":(test_audio_file_path,audio_file,'audio/wav')}
    
    response=requests.post(URL,files=values)
    
    data=response.json()
    
    print(f" Predicted Keyword : {data['keyword']} ")
    
    
    
    
    