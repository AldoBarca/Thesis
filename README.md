# Few-Shot Learning for Bio-Acoustic Event Detection

Master's thesis (Computer Engineering, University of Catania — A.Y. 2023/2024) on detecting and localizing animal sounds in long field recordings when you only have a handful of labelled examples per species.

The work follows the setup of the [DCASE 2024 Challenge, Task 5 — Few-shot Bioacoustic Event Detection](https://dcase.community/challenge2024/task-few-shot-bioacoustic-event-detection): given the first 5 annotated events of a species in a recording, find every other occurrence of the same sound in the rest of the audio.

## The idea

Labelled bioacoustic data is scarce and expensive — an expert has to sit and annotate hours of recordings. Few-shot learning sidesteps this by learning a general notion of "same sound / different sound" instead of a fixed set of classes.

The core model is a **Prototypical Network**: audio is projected into an embedding space, each class is represented by the mean (prototype) of its support examples, and query frames are classified by distance to those prototypes. Training uses a **dual objective** — cross-entropy on the classification head plus a triplet loss to pull same-class embeddings together and push different-class ones apart.

I compared three encoders behind the same prototypical head:

- **CNN trained from scratch** — a small convolutional baseline on Mel spectrograms.
- **AST** (Audio Spectrogram Transformer, `MIT/ast-finetuned-audioset-10-10-0.4593`) — a pretrained transformer, used as a frozen/fine-tuned feature extractor.
- **Wav2Vec2** — a self-supervised model that works directly on the raw waveform instead of spectrograms.

Localization is done on a **10 ms frame grid**: each frame gets a positive/negative prediction, adjacent positives are merged into events, and the predicted onset/offset are compared against ground truth.

## Repository layout

```
Thesis_code.ipynb   Main notebook — data prep, models, training, inference, evaluation
Json_files/         Annotations in DCASE format (train_data.json, val_data.json)
inference_stats_pth/  Saved evaluation results per encoder (torch .pth dicts)
All_Stats.xlsx      Aggregated metrics across runs
output.xlsx         Per-file metric export
Documents/          Reference papers (Prototypical Networks, DCASE winning systems)
Barca_Aldo_*.pptx   Thesis defense slides
Few-Shot ....pdf    Full thesis document
```

The notebook is organized top to bottom into clear sections: *Initialization → Utilities → Support/Query set creation → Prototypical Network → Training → Inference → Metrics*. You can read it in that order.

## Data

The audio itself (the DCASE **Development Set**, several GB of `.wav` files) is **not** in this repo — it lives outside version control. Only the JSON annotations are tracked here.

To reproduce:

1. Download the Development Set from the [DCASE Task 5 page](https://dcase.community/challenge2024/task-few-shot-bioacoustic-event-detection).
2. Point the path variables in the notebook (Training/Validation audio directories) at wherever you unpacked it.

## Running it

Tested with **Python 3.12**. Main dependencies:

```
torch  torchaudio  transformers  datasets
librosa  soundfile  mir_eval  scikit-learn  imbalanced-learn
numpy  scipy  pandas  matplotlib  openpyxl  tqdm
```

A GPU is strongly recommended (AST and Wav2Vec2 are heavy; the code uses mixed precision via `torch.cuda.amp`).

1. Open `Thesis_code.ipynb` in Jupyter or VS Code.
2. In the *Paths variables* section, set `drive = 0` for a local run (or `drive = 1` if you're mounting Google Drive on Colab) and edit the directory paths to match your machine.
3. Run the cells section by section — spectrogram/episode generation first, then training, then inference.

Because the encoders are expensive to train, intermediate artifacts (episodes, trained models, per-file stats) are cached to disk as `.pth` files so you don't recompute them every run.

## Results

Full numbers, ablations and discussion are in the [thesis PDF](Few_Shot_Learning_Models_for_Classification_and_Localization_of.pdf) and the [slides](Barca_Aldo_1000050457.pptx). Saved per-file precision/recall/F1 for each encoder are in `inference_stats_pth/` and summarized in `All_Stats.xlsx`.

Short version: performance varies a lot across recordings — some files are handled well, others (very sparse or noisy events) remain hard, which is consistent with the difficulty reported by the DCASE baselines and confirms that data scarcity is the real bottleneck the few-shot setup is trying to attack.

## References

- Snell et al., *Prototypical Networks for Few-shot Learning*, NIPS 2017 — `Documents/prototypical_networks_nips_2017.pdf`
- Moummad et al., DCASE 2023 IMT system — `Documents/DCASE2023_Moummad_IMT_t5.pdf`
- DCASE 2022/2023 winning systems — `Documents/Winner challenge_2023/`

## Author

**Aldo Barca** — MSc Computer Engineering, University of Catania.
Supervisor: Prof.ssa Daniela Giordano. Co-supervisors: Dr. Salvatore Calcagno, Dr. Simone Carnemolla.
