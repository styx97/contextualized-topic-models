from signal import default_int_handler
import configargparse
import shutil
from pathlib import Path

import json
import scipy
import numpy as np
from sentence_transformers import SentenceTransformer

from contextualized_topic_models.models.ctm import CombinedTM
from contextualized_topic_models.datasets.dataset import CTMDataset


def load_vocab(vocab_path):
    """
    Load the vocab json and return it in index2token format
    """

    with open(vocab_path) as f:
        vocab = json.load(f)

    return {k: v for v, k in vocab.items()}


def load_jsonl_dataset(text_path, jsonl_text_key="text"):
    """
    Load text dataset from jsonl to generate document embeddings
    """
    dataset = []
    with open(text_path) as f:
        for line in f:
            doc = json.loads(line.rstrip("\n"))
            text = " ".join(doc[jsonl_text_key].split()[:600]) # gets truncated at 512 anyway
            dataset.append(text)

    return dataset


def prepare_dataset(
    model_path="paraphrase-distilroberta-base-v1",
    dtm_path="train.dtm.npz",
    text_path="train.metadata.jsonl",
    embeds_path=None,
    vocab_path="vocab.json",
    jsonl_text_key="text",
    batch_size=200,
):
    """
    Prepare the dataset for training ctm using their own dataset class
    Note:
        1. The bow part of the model takes in data from our preprocessed input
        2. The contextual part of the model simply runs sentence transformer on the documents
            (well, the first 512 tokens of it)

    """

    bow = scipy.sparse.load_npz(dtm_path)
    print(f"Shape of bow - {bow.shape}\n")
    texts = load_jsonl_dataset(text_path, jsonl_text_key)
    idx2token = load_vocab(vocab_path)

    if embeds_path.exists():
        print("Loaded embeddings")
        embeddings = np.load(embeds_path)
    else:
        print("Creating embeddings")
        model = SentenceTransformer(model_path)
        embeddings = np.array(
            model.encode(texts, show_progress_bar=True, batch_size=batch_size)
        )
        Path(embeds_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(embeds_path, embeddings)

    assert bow.shape[0] == len(texts) == embeddings.shape[0]

    return CTMDataset(embeddings, bow, idx2token)


if __name__ == "__main__":
    parser = configargparse.ArgParser(
        description="parse args",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
    )

    parser.add("--run_embeddings_only", action="store_true", default=False)
    parser.add("-c", "--config", is_config_file=True, default=None)
    parser.add("--input_dir", required=True, default=None)
    parser.add("--output_dir", required=True, default=None)
    parser.add(
        "--jsonl_text_key", default="text", help="The key that has the document in it"
    )

    parser.add("--dtm_path", default="train.dtm.npz")
    parser.add("--text_path", default="train.metadata.jsonl")
    parser.add("--embeds_path", default=None)
    parser.add("--vocab_path", default="vocab.json")
    parser.add("--topic_word_init_path", default=None)
    parser.add("--topic_word_prior_path", default=None)
    parser.add("--embeddings_model", default="paraphrase-distilroberta-base-v1")


    parser.add("--model_type", default="prodLDA", help="ProdLDA or LDA")
    parser.add("--inference_type", default=None)

    parser.add(
        "--contextual_size",
        default=768,
        type=int,
        help="dimension of input that comes from the BERT embeddings",
    )
    parser.add(
        "--hidden_size", default=100, type=int, help="Duplicated to (hidden_size, hidden_size)"
    )
    parser.add("--num_topics", default=50, type=int)  # n_components
    parser.add("--dropout", default=0.2, type=float)

    parser.add("--batch_size", default=64, type=int)
    parser.add("--num_epochs", default=100, type=int)
    parser.add("--learning_rate", default=2e-3, type=float)
    parser.add("--momentum", default=0.99, type=float)

    parser.add("--n_samples", default=20, type=int, help="samples to use in theta estimate")
    # unique
    parser.add(
        "--no_learn_priors",
        action="store_false",
        dest="learn_priors",
        default=True,
        help="make priors a learnable parameter (default True)",
    )
    parser.add(
        "--reduce_on_plateau",
        action="store_true",
        default=False,
        help="reduce learning rate by 10x on plateau of 10 epochs",
    )
    parser.add("--solver", default="adam", help="adam or sgd")

    parser.add(
        "--run_seeds",
        default=[42],
        type=int,
        nargs="+",
        help="Seeds to use for each run",
    )
    parser.add("--gpu", action="store_true", default=False, help="whether to use cuda")
    parser.add(
        "--jit", action="store_true", default=False, help="whether to use PyTorch jit"
    )
    args = parser.parse_args()

    if args.embeds_path is None: 
        prefix = Path(args.dtm_path).stem.rstrip(".dtm") #train/test
        embeds_path = Path(args.input_dir, "embeddings", args.embeddings_model, f"{prefix}.embeds.npy")

    
    # organize input paths
    base_input_dir = Path(args.input_dir)
    dtm_path = base_input_dir / args.dtm_path
    text_path = base_input_dir / args.text_path
    vocab_path = base_input_dir / args.vocab_path
    

    # run the model
    # train
    dataset = prepare_dataset(
        model_path=args.embeddings_model,
        dtm_path=dtm_path,
        vocab_path=vocab_path,
        text_path=text_path,
        embeds_path=embeds_path,
        jsonl_text_key=args.jsonl_text_key,
    )

    if args.run_embeddings_only:
        exit()

    # print("prepared dataset")
    hidden_sizes = (args.hidden_size, args.hidden_size)
    ctm = CombinedTM(
        bow_size=dataset.X_bow.shape[1],
        contextual_size=args.contextual_size,
        n_components=args.num_topics,
        model_type="prodLDA",
        hidden_sizes=hidden_sizes,
        activation="softplus",
        dropout=args.dropout,
        learn_priors=args.learn_priors,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        momentum=args.momentum,
        solver=args.solver,
        num_epochs=args.num_epochs,
        reduce_on_plateau=args.reduce_on_plateau,
        loss_weights=None,
    )

    ctm.fit(dataset)  # run the model
    #from IPython import embed; embed()
    #print("IPYTHON SHOULD HAVE BEEN HIT BY NOW")

    # make sure the output directories exist
    output_dir = Path(args.output_dir)
    topics_path = output_dir / "topics.txt"
    output_dir.mkdir(exist_ok=True, parents=True)
    beta_path = output_dir / "beta.npy"
    train_theta_path = output_dir / "train.theta.npy"

    topic_word_dist = ctm.get_topic_word_distribution()
    doc_topic_dist = ctm.training_doc_topic_distributions # produced at end of run
    topics_list = ctm.get_topic_lists(k=100)

    with open(topics_path, 'w') as f: 
        for elem in topics_list: 
            s = " ".join(elem)
            f.write(f"{s}\n")

    # TODO: create the topics.txt file

    ctm.save(models_dir=output_dir)
    np.save(beta_path, topic_word_dist)
    np.save(train_theta_path, doc_topic_dist)