from email.mime import base
from pathlib import Path 
import warnings
import shutils 
import configargparse 
import scipy 
from contextualized_topic_models.models.ctm import CombinedTM
from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation
from contextualized_topic_models.datasets.dataset import CTMDataset
#from contextualized_topic_models.utils.preprocessing import WhiteSpacePreprocessing
from sentence_transformers import SentenceTransformer
import json 
import numpy as np 

def load_vocab(vocab_path): 
    """
    Load the vocab json and return it in index2token format
    """
    
    with open(vocab_path) as f: 
        vocab = json.load(f)
    
    return {k:v for v, k in vocab.items()}

def load_dataset(dataset_path, jsonl_text_key="text"): 
    """
    Load dataset from jsonl for loading into the CTM
    """
    dataset = []
    with open(dataset_path) as f: 
        for line in f: 
            doc = json.loads(line.rstrip("\n"))
            text = " ".join(doc[jsonl_text_key].split()[:600])  # after 512 will be truncated anyway
            dataset.append(text)

    return dataset
        

def prepare_dataset(model_path="paraphrase-distilroberta-base-v1", 
    train_path='train.dtm.npz', 
    vocab_path='vocab.json', 
    dataset_path='train.metadata.jsonl', 
    embeds_path='train.embeds.npy', 
    batch_size=200):
    """
    Prepare the dataset for training ctm using their own dataset class
    Note: 
        1. The bow part of the model takes in data from our preprocessed input 
        2. The contextual part of the model simply runs sentence transformer on the documents
            (well, the first 512 tokens of it)
    
    """
    
    bow = scipy.sparse.load_npz(train_path)
    print(f"Shape of bow - {bow.shape}\n")
    texts = load_dataset(dataset_path)
    idx2token = load_vocab(vocab_path)
    
    if embeds_path.exists(): 
        print("Found embedding vectors")
        with open(embeds_path, 'rb') as f: 
            embeddings = np.load(f)
            print("loaded embedding vectors")

    else : 
        print("Creating Embedding File")
        model = SentenceTransformer(model_path)
        embeddings = np.array(model.encode(texts, show_progress_bar=True, batch_size=batch_size))
        with open(embeds_path, 'wb') as f:
            np.save(f, embeddings) 
    
    
    assert(bow.shape[0] == len(texts) == embeddings.shape[0])
    
    return CTMDataset(embeddings, bow, idx2token)



if __name__ == "__main__": 
    parser = configargparse.ArgParser(
        description="parse args", 
        config_file_parser_class=configargparse.YAMLConfigFileParser
    )

    parser.add("--run_embeddings_only", action='store_true', default=False)
    parser.add("-c", "--config", is_config_file=True, default=None)
    parser.add("--input_dir", required=True, default=None) 
    parser.add("--output_dir", required=True, default=None)
    parser.add("--temp_output_dir", default=None, help="Temporary model storage during run, when I/O bound")
    parser.add("--jsonl_text_key", default="text", help="The key that has the document in it")

    parser.add("--train_path", default="train.dtm.npz")
    parser.add("--eval_path", default="val.dtm.npz")
    parser.add("--vocab_path", default="vocab.json")
    parser.add("--train_dataset_path", default="train.metadata.jsonl")
    parser.add("--topic_word_init_path", default=None)
    parser.add("--topic_word_prior_path", default=None)

    parser.add("--model_type", default="prodLDA", help="ProdLDA or LDA")
    parser.add("--inference_type",default=None)

    parser.add("--bow_size", type=int, default=None, help="dimension of input bow")
    parser.add("--contextual_size", default=768, type=int, help="dimension of input that comes from the BERT embeddings")
    parser.add("--hidden_sizes", default=(100, 100), type=tuple, help="length = n_layers")
    parser.add("--num_topics", default=50, type=int) # n_components
    parser.add("--dropout", default=0.2, type=float)
    
    parser.add("--batch_size", default=64, type=int)
    parser.add("--num_epochs", default=100, type=int)
    parser.add("--learning_rate", default=2e-3, type=float)
    parser.add("--momentum", default=0.99, type=float)
    
    #unique
    parser.add("--no_learn_priors", action='store_false', dest="learn_priors", default=True, help="make priors a learnable parameter (default True)")
    parser.add("--reduce_on_plateau", action='store_true', default=False, help="reduce learning rate by 10x on plateau of 10 epochs")
    parser.add("--cpu_count", default=1, type=int, help="number of data reader workers")
    parser.add("--solver", default="adam", help="adam or sgd")

    parser.add("--run_seeds", default=[42], type=int, nargs="+", help="Seeds to use for each run")
    parser.add('--gpu', action='store_true', default=False, help='whether to use cuda')
    parser.add('--jit', action='store_true', default=False, help='whether to use PyTorch jit')
    args = parser.parse_args()

    #check if the sentence encodings are saved in the input, else save them 
    base_input_dir = Path(args.input_dir)
    train_path = base_input_dir / args.train_path 
    vocab_path = base_input_dir / args.vocab_path 
    dataset_path = base_input_dir / args.train_dataset_path
    
    embeds_path = base_input_dir / "train.embeds.npy"
    
    #make sure the output directories exist 
    base_output_dir = Path(args.output_dir)
    base_output_dir.mkdir(exist_ok=True, parents=True)
    beta_path = base_input_dir / "beta.npy"
    train_theta_path = base_input_dir / "train.theta.npy"
    test_theta_path = base_input_dir / "test.theta.npy"


    # if len(args.run_seeds) == 1 : 
    #     output = base_output_dir 
    # else : 
    #     output_dir = Path(base_output_dir, str(seed))
    #     output_dir.mkdir(exist_ok=True, parents=True)

    run_dir = str(args.output_dir)
    if args.temp_output_dir:
        run_dir = Path(args.temp_output_dir, str(np.random.randint(1000)))

    # run the model 
    # train
    dataset = prepare_dataset(
        model_path = "paraphrase-distilroberta-base-v1", 
        train_path=train_path,
        vocab_path=vocab_path,
        dataset_path=dataset_path,
        embeds_path=embeds_path
    )

    if args.run_embeddings_only: 
        exit()

    #print("prepared dataset")
    ctm = CombinedTM(
        bow_size=dataset.X_bow.shape[1],
        contextual_size=args.contextual_size,
        n_components=args.num_topics,
        num_epochs=args.num_epochs,
        )
    
    
    ctm.fit(dataset) # run the model
    topic_word_dist = ctm.get_topic_word_distribution()
    doc_topic_dist = ctm.get_doc_topic_distribution(dataset)
    
    ctm.save(models_dir=args.output_dir)
    np.save(train_theta_path, doc_topic_dist)
    np.save(beta_path, topic_word_dist)

    if args.temp_output_dir:
        shutil.copytree(run_dir, output_dir, dirs_exist_ok=True)
        shutil.rmtree(run_dir)

