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

print(conta_eventi_set(support_set))
print(conta_eventi_set(query_set))



def create_support_set_training():
   
    support_set_train=torch.empty(conta_classi_set(support_set),3,3,3)
    print(support_set_train.shape)
    for classe,file in support_set.items():
        campioni_positivi_classe=torch.empty(3,3)#[]
        for audio,events in file.items():
            audio_path=os.path.join(args.traindir,audio)
            audio_loaded, sr=librosa.load(audio_path)
            for evento in events:
                    start_campione=int(evento['start']*sr)
                    end_campione=int(evento['end']*sr)
                    
                    audio_evento=audio_loaded[start_campione:end_campione]  #frammento di audio contenente solo l'evento positivo
                    lunghezza_frame=int(0.01*sr)

                    #scomposizione in frames del frammento di audio dell'evento positivo, diventa un tensore di tipo [180][220]
                    #il 180 varia in base alla lunghezza del frammento audio 
                    frames=librosa.util.frame(audio_evento, frame_length=lunghezza_frame,hop_length=lunghezza_frame).T    #l'hop lenght lo faccio intanto senza sovrapposizioni, poi vediamo
                    #campioni_positivi_classe.append(frames)

                    torch.cat(campioni_positivi_classe,frames)
                    
                    
                    
        indice_classe=class_dictionary[classe]
        support_set_train[indice_classe]=campioni_positivi_classe
        
    return support_set_train


support_set_train=create_support_set_training()
print(support_set_train)




#stft = torchlibrosa.stft.Spectrogram(n_fft=n_fft, hop_length=hop_length, win_length=win_length, power=2.0)