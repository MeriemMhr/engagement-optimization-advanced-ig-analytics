from gensim import corpora, models
from gensim.models.coherencemodel import CoherenceModel

def perform_lda(labels_list, num_topics=10):
    """
    Perform Latent Dirichlet Allocation (LDA) for topic modeling.

    Args:
        labels_list (list of list of str): List of documents, each represented as a list of words.
        num_topics (int, optional): Number of topics to generate. Default is 10.

    Returns:
        tuple: Contains the LDA model, dictionary, corpus, and coherence score.
    """
    # Create a dictionary from the labels_list. Each unique word gets an ID.
    dictionary = corpora.Dictionary(labels_list)
    # Convert list of words in documents into the bag-of-words (BoW) format.
    corpus = [dictionary.doc2bow(text) for text in labels_list]

    # Set up the LDA model with the corpus and dictionary, specifying parameters like the number of passes and alpha.
    lda_model = models.LdaModel(
        corpus=corpus, id2word=dictionary, num_topics=num_topics,
        random_state=100, update_every=1, passes=10, alpha='auto', per_word_topics=True
    )
    # Calculate model coherence to evaluate its performance.
    coherence_model_lda = CoherenceModel(model=lda_model, texts=labels_list, dictionary=dictionary, coherence='c_v')
    coherence = coherence_model_lda.get_coherence()

    return lda_model, dictionary, corpus, coherence

def find_optimal_topics(labels_list, start=2, limit=20, step=3):
    """
    Determines the optimal number of topics for LDA using a range of possible topic counts.

    Args:
        labels_list (list of list of str): List of documents, each represented as a list of words.
        start (int): Starting number of topics to test.
        limit (int): Maximum number of topics to test.
        step (int): Increment of topics in each iteration.

    Returns:
        tuple: Contains list of models, their coherence values, and the optimal number of topics.
    """
    coherence_values = []
    model_list = []
    topic_numbers = []

    # Iterate over possible topic counts to find the optimal number by maximizing coherence.
    for num_topics in range(start, limit, step):
        model, _, _, coherence = perform_lda(labels_list, num_topics=num_topics)
        model_list.append(model)
        coherence_values.append(coherence)
        topic_numbers.append(num_topics)

    # Identify the model with the highest coherence score and determine the optimal number of topics.
    max_coherence_idx = coherence_values.index(max(coherence_values))
    optimal_num_topics = topic_numbers[max_coherence_idx]

    return model_list, coherence_values, optimal_num_topics
