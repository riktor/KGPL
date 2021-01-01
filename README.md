# KGPL

This repository is the implementation of KGPL ([arXiv](https://arxiv.org/abs/2011.05061)):

> Alleviating Cold-Start Problems in Recommendation through Pseudo-Labelling over Knowledge Graph
Riku Togashi, Mayu Otani, Shin'ichi Satoh.  
In Proceedings of The 14th ACM International Conference on Web Search and Data Mining (WSDM 2021)  

KGPL is a method to alleviate cold-start problems in personalised recommendation tasks.
It expoits users' implicit feedback and a knowledge graph to enhance the sampling for pseudo-labelling.
The codes are mainly based on the implementation of KGNN-LS by authors ([here](https://github.com/hwwang55/KGNN-LS)).

### Files

- `data/`
  - `movie/`    
    - `item_index2entity_id.txt`: the mapping from item indices in the raw rating file to entity IDs in the KG;
    - `kg.txt`: knowledge graph file;
	- `ratings.txt`: raw rating file of MovieLens-1M;
  - `music/`
    - `item_index2entity_id.txt`: the mapping from item indices in the raw rating file to entity IDs in the KG;
    - `kg.txt`: knowledge graph file;
    - `user_artists.dat`: raw rating file of Last.FM;
  - `book/`
    - `item_index2entity_id.txt`: the mapping from item indices in the raw rating file to entity IDs in the KG;
    - `kg.txt`: knowledge graph file;
	- `BX-Book-Ratings.csv`: raw rating file of Book-Crossing;
- `conf/`:
  - `dataset`: "dataset_name.yaml" contains the path configurations;
  - `model`: "model_name.yaml" contains the model configurations;
  - `config.yaml`: default config file;
  - `preprocess.yaml`: default config file for path finding;
- `preprocess`:
  - `preprocess.py`: preprocessing for raw rating files;
  - `split_data.py`: script for splitting data;
  - `make_path_list.py`: script for path finding;
- `models`:
  - `train.py`: the implementation of training process;
  - `aggregators.py`: the aggregator of KGNN-LS;
  - `kgpl.py`: the implementation of training process;
  - `utils.py`: utilities;
  
### Running the code
- Music  
  ```
  $ python preprocess/preprocess.py -d music
  $ python preprocess/make_path_list.py lp_depth=6 dataset=music kg_path=data/music/kg_final.npy rating_path=data/music/ratings_final.npy num_neighbor_samples=32
  $ python preprocess/split_data.py dataset=music rating_path=data/music/ratings_final.npy
  $ python main.py model=kgplcot_music log.experiment_name=music optimize.n_epochs=40 dataset=music evaluate.user_num_topk=2000 log.show_loss=True
  ```

- Movie  
  ```
  $ python preprocess/preprocess.py -d movie
  $ python preprocess/make_path_list.py lp_depth=5 dataset=movie kg_path=data/movie/kg_final.npy rating_path=data/movie/ratings_final.npy num_neighbor_samples=32
  $ python preprocess/split_data.py dataset=movie rating_path=data/movie/ratings_final.npy
  $ python main.py model=kgplcot_movie log.experiment_name=movie optimize.n_epochs=40 dataset=movie evaluate.user_num_topk=2000 log.show_loss=True
  ```

- Book  
  ```
  $ python preprocess/preprocess.py -d book
  $ python preprocess/make_path_list.py lp_depth=6 dataset=book kg_path=data/book/kg_final.npy rating_path=data/book/ratings_final.npy num_neighbor_samples=8
  $ python preprocess/split_data.py dataset=book rating_path=data/book/ratings_final.npy
  $ python main.py model=kgplcot_book log.experiment_name=book optimize.n_epochs=40 dataset=book evaluate.user_num_topk=2000 log.show_loss=True
  ```
  
You can suppress the logs for each training step (e.g. losses and gradients) by setting `log.show_loss=False`.
See `config.yaml` for further detailed configurations.
