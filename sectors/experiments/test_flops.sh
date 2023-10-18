python test_flops.py --batch_size 1 --load_in_8bit
python test_flops.py --batch_size 1 --load_in_8bit --head ch

python test_flops.py --batch_size 1 --load_in_8bit --model_name bigscience/bloom-1b7 --batch_size 1
python test_flops.py --batch_size 1 --load_in_8bit --model_name bigscience/bloom-1b7 --head ch --batch_size 4