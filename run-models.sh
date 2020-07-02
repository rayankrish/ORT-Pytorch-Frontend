echo "Starting Torch and ORT models with given weights. Will save model weights"
echo "----- Torch Model -----"
python mnist-torch-mod.py --set-weights --save-full
echo "----- ORT Model -----"
python mnist-ort.py --set-weights --save-full
echo "===== DONE ====="

