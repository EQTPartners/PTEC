python gzip_classification.py

python embedding_proximity/vector_similarity.py
python embedding_proximity/vector_similarity.py --augmented augmented
python embedding_proximity/vector_similarity.py --model_name bigscience/bloom-1b7
python embedding_proximity/vector_similarity.py --model_name bigscience/bloom-1b7 --augmented augmented

python embedding_proximity/vector_similarity.py --type RadiusNN
python embedding_proximity/vector_similarity.py --type RadiusNN --augmented augmented
python embedding_proximity/vector_similarity.py --type RadiusNN --model_name bigscience/bloom-1b7
python embedding_proximity/vector_similarity.py --type RadiusNN --model_name bigscience/bloom-1b7 --augmented augmented

python embedding_proximity/classification_head/classification_head.py
python embedding_proximity/classification_head/classification_head.py --augmented augmented
python embedding_proximity/classification_head/classification_head.py --model_name bigscience/bloom-1b7
python embedding_proximity/classification_head/classification_head.py --model_name bigscience/bloom-1b7 --augmented augmented
