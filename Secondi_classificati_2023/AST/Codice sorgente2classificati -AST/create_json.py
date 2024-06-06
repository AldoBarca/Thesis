import os
import json
import glob
from args import args




def create_wav_csv_mapping(root_directory):
    # Dizionario per memorizzare la mappatura tra file WAV e CSV
    wav_csv_mapping = {}

    # Itera attraverso tutte le cartelle e i file nella directory principale
    csv_files = [f for f in glob.glob(os.path.join(root_directory, '*/*.csv'))]
    wav_files = [f for f in glob.glob(os.path.join(root_directory, '*/*.wav'))]

        # Crea un set di nomi base (senza estensione) per facilitare la ricerca dei file corrispondenti
    wav_base_names = {os.path.splitext(f)[0] for f in wav_files}
    csv_base_names = {os.path.splitext(f)[0] for f in csv_files}

        
    common_names = wav_base_names.intersection(csv_base_names)
    for name in common_names:
        wav_csv_mapping[f"{name}.wav"] = f"{name}.csv"

    return wav_csv_mapping

def save_mapping_to_json(mapping, json_file_path):
    with open(json_file_path, 'w') as json_file:
        json.dump(mapping, json_file, indent=4)

json_file_path = 'wav_csv_mapping.json'
#salva il json
wav_csv_mapping = create_wav_csv_mapping(args.traindir)
save_mapping_to_json(wav_csv_mapping, json_file_path)

print(f"Mappatura salvata in {json_file_path}")
