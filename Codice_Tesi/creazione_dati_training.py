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


def return_max_num_eventi(set):
    eventi=conta_eventi_set(set)
    i=0
    for classe,num_eventi in eventi.items():
        if eventi[classe]>i:
            i=eventi[classe]
    return i

print(conta_eventi_set(support_set))
print(conta_eventi_set(query_set))
print(return_max_num_eventi(support_set))
print(return_max_num_eventi(query_set))

#support_vector:[Classe,Evento,]
#query_label=num_frames


def ritorna_numero_frame_minime(set,durata_frame_in_secondi,frame_length,hop_length):
    num_frames=0
    for classe,file in set.items():
        for audio,events in file.items():
             for evento in events:
                intervallo=evento['end']-evento['start']
                sovrapposizione=hop_length/frame_length #quanto effettivamente le frame si sovrappongono l'un l'altra
                num_frames_evento=intervallo/(durata_frame_in_secondi*sovrapposizione)              #durata di 0.01 se frame=10ms.
                
                if (num_frames==0):
                    num_frames=num_frames_evento
                elif (num_frames>num_frames_evento):
                    num_frames=num_frames_evento
    return num_frames
numero_frames_support=ritorna_numero_frame_minime(support_set,0.01,1,1)
numero_frames_query=ritorna_numero_frame_minime(query_set,0.01,1,1)
"""

Si hanno due soluzioni per il support set:
1.Considero il massimo di eventi del support set tra i vari audio, ergo 25. 
A)Devo però riempire di eventi Negativi anche il support set.Concetto che non sono sicuro abbia senso per un support set, che dovrebbe mostrarci 
solo eventi positivi per definizione.
B)In alternativa effettuare data augmentation che partendo da un certo numero di campioni arrivi a 25. 

2.Considero il minimo (5) eventi. 5 eventi sarebbe interessante anche perche alleneremmo il modello a lavorare con un support set di k=5 esempi come
durante l'evaluation, ma dovremmo non usare di fatto molti audio, aspetto che potrebbe essere controproducente in fase di training.
"""

def create_support_set_training1B(set):
    classi_set=conta_classi_set(set)
    max_numero_eventi=return_max_num_eventi(set)
    num_min_frame=ritorna_numero_frame_minime(support_set)
    support_set_train=torch.empty(classi_set,max_numero_eventi,num_min_frame,220)

    for classe,file in support_set.items():
        id_evento=0

        for audio,events in file.items():
            audio_path=os.path.join(args.traindir,audio)
            audio_loaded, sr=librosa.load(audio_path)
            for evento in events:
                start_campione=int(evento['start']*sr)
                end_campione=int(evento['end']*sr)
                audio_evento=audio_loaded[start_campione:end_campione]


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


support_set_train=create_support_set_training()
print(support_set_train)




#stft = torchlibrosa.stft.Spectrogram(n_fft=n_fft, hop_length=hop_length, win_length=win_length, power=2.0)