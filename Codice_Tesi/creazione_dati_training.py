"""Pensare ad una funzione che importi l’audio, lo estragga in segmenti. Va capito come trattare i segmenti come framing. Vogliamo tensori.

Struttura: Non facciamo trasformazioni. Vogliamo solo ottimizzare come prendere i primi 5 campioni di ogni audio del suo support.


Dim1 varie classi, dim 2: uguale a 5+5+3+5, dim3: lenght dei segmenti. Dim variabili.
Creo un tensore per query e 1 tensore per support.
Per ogni class creo:
Support sets classe 1,n: 5 eventi positive
Query sets classe 1,n: n-5 eventi

Support_set_shape_finale:
[n_classi,num_eventi,num_frames,num_campioni]
"""
#audio_files:
#nome_classe: [file][start,fine]
#support[classe][]
#vettore[nome_audio] restituisce start e end
import torch
import torchlibrosa
import os
import librosa
from args import args
import json
import numpy as np
import random
import math

class_dictionary={}




"""
ouput del support set in formato dictionary con chiave nome_classe:
nome_classe:{audio1:[{start,end},{start,end}]}


"""

def get_support_set():
    class_list=[]
    csv_files={}
    audio_files={}
    
    with open(args.train_json, 'r') as file:
        dati = json.load(file)

    for classe, value in dati.items():
        class_list+={classe}
        i=0
    for classe in class_list:
        class_dictionary[classe]=i
        i=i+1


    for classe in class_list:
        for csv in dati[classe]:
            if classe in csv_files:
                csv_files[classe].append(csv)
            else:
                csv_files[classe] = [csv]
            if(dati[classe][csv]["Support"]!={}):   
                              #verifico che ci siano eventi positivi in quell'audio della classe 
                if classe not in audio_files:
                    audio_files[classe] = {}
                  
                
                if dati[classe][csv]["Audio"] not in audio_files[classe]:
                    audio_files[classe][dati[classe][csv]["Audio"]] = {}
                #audio_files[classe][dati[classe][csv]["Audio"]]=dati[classe][csv]["Audio"]["Supports"] Se volessi un formato diverso con anche i nomi degli eventi(seg0,1 ect...)
                lista_start_e_end_time_eventi=[]
                for evento in dati[classe][csv]["Support"]:
                    lista_start_e_end_time_eventi.append(dati[classe][csv]["Support"][evento])
               
                audio_files[classe][dati[classe][csv]["Audio"]]=lista_start_e_end_time_eventi
   
  
    return audio_files




def get_query_set():
    class_list=[]
    csv_files={}
    audio_files={}
    with open(args.train_json, 'r') as file:
        dati = json.load(file)
    for classe, value in dati.items():
            class_list+={classe}
        #print(len(class_list))
    for classe in class_list:
        for csv in dati[classe]:
            if classe in csv_files:
                csv_files[classe].append(csv)
            else:
                csv_files[classe] = [csv]
            if(dati[classe][csv]["Query"]!={}):   
                            #verifico che ci siano eventi positivi in quell'audio della classe 
                if classe not in audio_files:
                    audio_files[classe] = {}
                
                if dati[classe][csv]["Audio"] not in audio_files[classe]:
                    audio_files[classe][dati[classe][csv]["Audio"]] = {}
                #audio_files[classe][dati[classe][csv]["Audio"]]=dati[classe][csv]["Audio"]["Supports"] Se volessi un formato diverso con anche i nomi degli eventi(seg0,1 ect...)
                lista_start_e_end_time_eventi=[]
                for evento in dati[classe][csv]["Query"]:
                    lista_start_e_end_time_eventi.append(dati[classe][csv]["Query"][evento])

                audio_files[classe][dati[classe][csv]["Audio"]]=lista_start_e_end_time_eventi
    return audio_files

support_set=get_support_set()  
query_set=get_query_set()
#print(support_set)


def conta_classi_set(set):
    i=0
    for classe,file in set.items():
        i=i+1
    return i
def conta_eventi_set(set):
    eventi_classi={}
    for classe,file in set.items():
        i=0
        for audio,events in file.items():
            for evento in events:
                i=i+1
        eventi_classi[classe]=i

    return eventi_classi


def return_max_num_eventi(set):
    eventi=conta_eventi_set(set)
    i=0
    for classe,num_eventi in eventi.items():
        if eventi[classe]>i:
            i=eventi[classe]
    return i



#support_vector:[Classe,Evento,]
#query_label=num_frames


def ritorna_numero_frame(set,durata_frame_in_secondi,frame_length=1,hop_length=1):
    num_frames_min=0
    num_frames_max=0
    for classe,file in set.items():
        for audio,events in file.items():
             for evento in events:
                intervallo=evento['end']-evento['start']
                sovrapposizione=hop_length/frame_length #quanto effettivamente le frame si sovrappongono l'un l'altra
                num_frames_evento=intervallo/(durata_frame_in_secondi*sovrapposizione)              #durata di 0.01 se frame=10ms.
                if(num_frames_evento>num_frames_max):
                    num_frames_max=num_frames_evento
                if (num_frames_min==0):
                    num_frames_min=num_frames_evento
                elif (num_frames_min>num_frames_evento):
                    num_frames_min=num_frames_evento
    return num_frames_max,num_frames_min




"""
Problema numero di eventi variabili.
Si hanno due soluzioni per il support set:
1.Considero il massimo di eventi del support set tra i vari audio, ergo 25. 
A)Devo però riempire di eventi Negativi anche il support set.Concetto che non sono sicuro abbia senso per un support set, che dovrebbe mostrarci 
solo eventi positivi per definizione.
B)In alternativa effettuare data augmentation che partendo da un certo numero di eventi arrivi a 25. 

2.Considero il minimo (5) eventi. 5 eventi sarebbe interessante anche perche alleneremmo il modello a lavorare con un support set di k=5 esempi come
durante l'evaluation, ma dovremmo non usare di fatto molti audio, aspetto che potrebbe essere controproducente in fase di training.



Problema eventi di durata variabile, ergo diverso numero di frame per evento.
Mi vengono per il momento 2 tipi di soluzioni:
1)Consideriamo la durata dell'evento più breve e rendiamo ogni evento di quella durata,risoluzione banale, tuttavia per il support set
l'evento più breve dura 2 frame e quello più lungo oltre 180 frames. Perdita di dati molto grave (rimaniamo con 1/90 dei dati)


2) Data augmentation che crei nuove frame partendo dalle precedenti. Cosi teoricamente possiamo rendere ogni evento della stessa durata di frame
uguale al numero di frame dell'evento più lungo.

                
                Va fatta la data augmentation per rendere costanti i numeri degli eventi, si potrebbe fare:
                1)Time_stretch      audio_evento_augmented=librosa.effects.time_stretch(audio_evento, rate)

                2)pitch_shift       audio_evento_augmented=librosa.effects.pitch_shift(audio_evento, rate)

                3)aggiunta_rumore   noise = np.random.randn(len(audio_evento))
                                    audio_evento_augmented=audio_evento+(0.05*noise)

                4)shift_time        shift=np.random.randint(len(audio_evento) * 0.2)
                                    audio_evento_augmented=np.roll(audio_evento,shift)

                5)amplificazione    ampl=2.0
                                    audio_evento_augmented=audio_evento*ampl

                6)random_crop       crop_len=int(len(audio_evento) * 0.8)
                                    start = random.randint(0, len(audio_evento) - crop_len)
                                    audio_evento_augmented=audio_evento[start:start+crop_len]
                                    
"""


def create_support_set_training1B(set):
    classi_set=conta_classi_set(set)
    max_numero_eventi=return_max_num_eventi(set)
    num_max_frame,num_min_frame=ritorna_numero_frame(set,0.01)
    num_max_frame=math.floor(num_max_frame)
    support_set_train=torch.empty(classi_set,max_numero_eventi,num_max_frame,220)

    for classe,file in support_set.items():
        id_classe=class_dictionary[classe]
        for audio,events in file.items():
            id_evento=1

            audio_path=os.path.join(args.traindir,audio)
            audio_loaded, sr=librosa.load(audio_path)

            for evento in events:
                id_frame=0

                start_campione=int(evento['start']*sr)
                end_campione=int(evento['end']*sr)
                audio_evento=audio_loaded[start_campione:end_campione]
                lunghezza_frame_campioni=int(0.01*sr)


                frames=librosa.util.frame(audio_evento, frame_length=lunghezza_frame_campioni,hop_length=lunghezza_frame_campioni).T 
                writable_copy=np.copy(frames)
                frames=torch.from_numpy(writable_copy)
                frames_evento=frames
                id_frame=id_frame+len(frames)
                if(id_frame==num_max_frame):
            
                    support_set_train[id_classe,id_evento]=frames_evento
                    print(support_set_train)


                if(id_frame>num_max_frame):
                    frames_evento=frames_evento[0:num_max_frame,:]
                    support_set_train[id_classe,id_evento]=frames_evento
                else:
                        while(id_frame<num_max_frame):
                            #data augmentation
                            shift=np.random.randint(len(audio_evento) * 0.2)
                            audio_evento_augmented=np.roll(audio_evento,shift)
                            noise = np.random.randn(len(audio_evento_augmented))
                            audio_evento_augmented=audio_evento_augmented+(0.05*noise)

                            frames=librosa.util.frame(audio_evento_augmented, frame_length=lunghezza_frame_campioni,hop_length=lunghezza_frame_campioni).T 
                            writable_copy=np.copy(frames)
                            frames=torch.from_numpy(writable_copy)

                            frames_evento=torch.cat((frames_evento,frames),dim=0)
                            id_frame=id_frame+len(frames)
                            if(id_frame==num_max_frame):
                                frames_evento=frames_evento[0:num_max_frame,:]
                                support_set_train[id_classe,id_evento]=frames_evento
                                break

                            if(id_frame>num_max_frame):
                                frames_evento=frames_evento[0:num_max_frame,:]
                                support_set_train[id_classe,id_evento]=frames_evento
                                break
                
                id_evento=id_evento+1
                print(id_evento)

            #Da qui in poi si riempiono di eventi "augmented" le classi con meno eventi

            while(id_evento!=max_numero_eventi):
                for evento in events:
                    id_frame=0
                    if(random.randint(0, 1)>0.9):    #per ogni evento ho il 40% di prendere l'evento.
                        continue
                    start_campione=int(evento['start']*sr)
                end_campione=int(evento['end']*sr)
                audio_evento=audio_loaded[start_campione:end_campione]

                #data augmentation

                shift=np.random.randint(len(audio_evento) * 0.2)
                audio_evento_augmented=np.roll(audio_evento,shift)
                noise = np.random.randn(len(audio_evento_augmented))
                audio_evento_augmented=audio_evento_augmented+(0.05*noise)
                lunghezza_frame_campioni=int(0.01*sr)
                crop_len=int(len(audio_evento_augmented) * 0.8)
                start = random.randint(0, len(audio_evento_augmented) - crop_len)
                audio_evento_augmented=audio_evento_augmented[start:start+crop_len]


                frames=librosa.util.frame(audio_evento, frame_length=lunghezza_frame_campioni,hop_length=lunghezza_frame_campioni).T 
                writable_copy=np.copy(frames)
                frames=torch.from_numpy(writable_copy)
                frames_evento=frames
                id_frame=id_frame+len(frames)
                if(id_frame==num_max_frame):
                 
                    support_set_train[id_classe,id_evento]=frames_evento
                    


                if(id_frame>num_max_frame):
                    frames_evento=frames_evento[0:num_max_frame,:]
                    support_set_train[id_classe,id_evento]=frames_evento
                else:
                        while(id_frame<num_max_frame):
                            #data_augmentation
                            shift=np.random.randint(len(audio_evento) * 0.2)
                            audio_evento_augmented=np.roll(audio_evento,shift)
                            noise = np.random.randn(len(audio_evento_augmented))
                            audio_evento_augmented=audio_evento_augmented+(0.05*noise)

                            frames=librosa.util.frame(audio_evento_augmented, frame_length=lunghezza_frame_campioni,hop_length=lunghezza_frame_campioni).T 
                            writable_copy=np.copy(frames)
                            frames=torch.from_numpy(writable_copy)

                            frames_evento=torch.cat((frames_evento,frames),dim=0)
                            id_frame=id_frame+len(frames)
                            if(id_frame==num_max_frame):
                                frames_evento=frames_evento[0:num_max_frame,:]
                                support_set_train[id_classe,id_evento]=frames_evento
                                break

                            if(id_frame>num_max_frame):
                                frames_evento=frames_evento[0:num_max_frame,:]
                                support_set_train[id_classe,id_evento]=frames_evento
                                break
                id_evento=id_evento+1
                print(id_evento)
        print("classe:{} finita"+classe)
        print(support_set_train)
        print(support_set_train.shape)

    
    return support_set_train
                    


                


def create_support_set_training():
   
    support_set_train=torch.empty(conta_classi_set(support_set),300,300,300)
   
    for classe,file in support_set.items():
        campioni_positivi_classe=torch.empty(3,3)
        for audio,events in file.items():
            audio_path=os.path.join(args.traindir,audio)
            audio_loaded, sr=librosa.load(audio_path)
            for evento in events:
                    start_campione=int(evento['start']*sr)
                    end_campione=int(evento['end']*sr)
                    
                    audio_evento=audio_loaded[start_campione:end_campione]  #frammento di audio contenente solo l'evento positivo
                    lunghezza_frame_campioni=int(0.01*sr) #220 campioni per frame

                    #scomposizione in frames del frammento di audio dell'evento positivo, diventa un tensore di tipo [180][220]
                    #il 180 varia in base alla lunghezza del frammento audio 
                    frames=librosa.util.frame(audio_evento, frame_length=lunghezza_frame_campioni,hop_length=lunghezza_frame_campioni).T    #l'hop lenght lo faccio intanto senza sovrapposizioni, poi vediamo
                    #campioni_positivi_classe.append(frames)
                    frames=torch.from_numpy(frames)
                    torch.stack(campioni_positivi_classe,frames)
                    
                    
                    
        indice_classe=class_dictionary[classe]
        support_set_train[indice_classe]=campioni_positivi_classe
        
    return support_set_train


support_set_train=create_support_set_training1B(support_set)
print(support_set_train)




#stft = torchlibrosa.stft.Spectrogram(n_fft=n_fft, hop_length=hop_length, win_length=win_length, power=2.0)