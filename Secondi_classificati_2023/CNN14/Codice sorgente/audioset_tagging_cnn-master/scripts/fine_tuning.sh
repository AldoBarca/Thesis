set MODEL_TYPE="Transfer_Cnn14"
set CHECKPOINT_PATH="Cnn14_mAP=0.431.pth"
set CUDA_VISIBLE_DEVICES=0 
python3 pytorch/finetune_template.py train ^
    --sample_rate=22050 ^
    --window_size=1024 ^
    --hop_size=320 ^
    --mel_bins=128 ^
    --fmin=50 ^
    --fmax=14000 ^
    --model_type=%MODEL_TYPE% ^
    --pretrained_checkpoint_path=%CHECKPOINT_PATH% ^
    --cuda
