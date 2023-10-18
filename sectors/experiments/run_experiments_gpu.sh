python embedding_proximity/generate_embeddings.py
python embedding_proximity/generate_embeddings.py --augmented augmented
python embedding_proximity/generate_embeddings.py --model_name bigscience/bloom-1b7 --batch_size 8
python embedding_proximity/generate_embeddings.py --model_name bigscience/bloom-1b7 --batch_size 8 --augmented augmented

python nshot/nshot.py
python nshot/nshot.py --model_name bigscience/bloom-1b7
python nshot/nshot.py --trie_search True
python nshot/nshot.py --trie_search True --model_name bigscience/bloom-1b7

python prompt_tuning/prompt_tune.py
python prompt_tuning/prompt_tune.py --augmented augmented
python prompt_tuning/prompt_tune.py --model_name bigscience/bloom-1b7 --batch_size 8 --interrupt_threshold 0.01
python prompt_tuning/prompt_tune.py --model_name bigscience/bloom-1b7 --batch_size 8 --interrupt_threshold 0.01 --augmented augmented

python prompt_tuning/prompt_tune.py --head ch --scheduler exponential
python prompt_tuning/prompt_tune.py --head ch --scheduler exponential --augmented augmented
python prompt_tuning/prompt_tune.py --model_name bigscience/bloom-1b7 --head ch --scheduler exponential --batch_size 8 --interrupt_threshold 0.01
python prompt_tuning/prompt_tune.py --model_name bigscience/bloom-1b7 --head ch --scheduler exponential --batch_size 8 --interrupt_threshold 0.01 --augmented augmented
