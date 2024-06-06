import os
import shutil

def unificatore_file_audio(root_directory):
    audios_directory = os.path.join(root_directory, "Audios")
    os.makedirs(audios_directory, exist_ok=True)
    for folder_name in os.listdir(root_directory):
        folder_path = os.path.join(root_directory, folder_name)
        
        if os.path.isdir(folder_path):
            
            for file_audio in os.listdir(folder_path):
                
                if file_audio.endswith('.wav'):
                    file_path = os.path.join(folder_path, file_audio)
                    destination_path=os.path.join(audios_directory, file_audio)
                    if not os.path.exists(destination_path):
                        shutil.copy(file_path, audios_directory)
                        print("File {} copiato con successo!".format(file_audio))


root = "E:/Tesi/Development_Set/Training_Set"  # Sostituire con il proprio percorso della directory principale
unificatore_file_audio(root)