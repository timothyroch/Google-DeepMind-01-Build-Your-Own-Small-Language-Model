import re

class SimpleWordTokenizer:
    """A simple word tokenizer that can be initialized with a corpus of texts
       or using a provided vocabulary list.

    The tokenizer splits the text sequence based on spaces,
    using the `encode` method to convert the text into a sequence of indices
    and the `decode` method to convert indices back into text.

    """

    def __init__(self, corpus: list[str], vocabulary: list[str] | None = None):
        """
        Initializes the tokenizer with texts in corpus or with a vocabulary.
        """

        if vocabulary is None:
            # Build the vocabulary from scratch.
            if isinstance(corpus, str):
                corpus = [corpus]

            # Convert text sequence to tokens.
            tokens = []
            for text in corpus:
                for token in self.space_tokenize(text):
                    tokens.append(token)

            # Create a vocabulary comprising of unique tokens.
            self.vocabulary = self.build_vocabulary(tokens)

        else:
            self.vocabulary = vocabulary

        # Size of vocabulary.
        self.vocabulary_size = len(self.vocabulary)

        # Create token-to-index and index-to-token mappings.
        self.token_to_index = {}
        self.index_to_token = {}
        # Loop through all tokens in the vocabulary. enumerate automatically
        # assigns a unique index to each token.
        for index, token in enumerate(self.vocabulary):
            self.token_to_index[token] = index
            self.index_to_token[index] = token

    def space_tokenize(self, text: str) -> list[str]:
        """
        Splits a given text on space into tokens.
        """

        # Use re.split such that multiple spaces are treated as a single
        # separator.
        return re.split(" +", text)

    def join_text(self, text_list: list[str]) -> str:
        """
        Combines a list of tokens into a single string, with tokens separated
           by spaces.
        """
        return " ".join(text_list)

    def build_vocabulary(self, tokens: list[str]) -> list[str]:
        """
        Create a vocabulary list from the list of tokens.
        """
        return sorted(list(set(tokens)))

    def encode(self, text: str) -> list[int]:
        """
        Encodes a text sequence into a list of indices.
        """

        # Convert tokens into indices.
        indices = []
        for token in self.space_tokenize(text):
            token_index = self.token_to_index.get(token)
            indices.append(token_index)

        return indices

    def decode(self, indices: int | list[int]) -> str:
        """
        Decodes a list (or single index) of integers back into tokens.
        """

        # If a single integer is passed, convert it into a list.
        if isinstance(indices, int):
            indices = [indices]

        # Map indices to tokens.
        tokens = []
        for index in indices:
            token = self.index_to_token.get(index)
            tokens.append(token)

        # Join the decoded tokens into a single string.
        return self.join_text(tokens)
