import re
import collections
import unicodedata


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip('\n')
        vocab[token] = index
    return vocab


def whitespace_tokenize(text):
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens

class Tokenizer(object):
    """Runs end-to-end tokenziation."""
    def __init__(self, vocab, do_lower_case=True, tokenize_chinese_chars=True, 
                token_start='[CLS]', token_end='[SEP]', max_inputs_chars_per_word=200):
        self._token_pad = '[PAD]'
        self._token_unk = '[UNK]'
        self._token_mask='[MASK]'
        self._token_start = token_start
        self._token_end = token_end
        self.basic_tokenizer = BasicTokenizer(do_lower_case, tokenize_chinese_chars)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab, self._token_unk, max_inputs_chars_per_word)
        self.token2id = vocab
        self.id2token = {v:k for k,v in vocab.items()}
        self.do_lower_case = do_lower_case
        for token in ['pad', 'unk', 'mask', 'start', 'end']:
            try:
                _token_id = self.token2id[getattr(self, '_token_%s' % token)]
                setattr(self, '_token_%s_id' % token, _token_id)
            except:
                pass
    def rematch(self, text, tokens):
        tmp, char_mapping = '', []
        for i, ch in enumerate(text):
            if self.do_lower_case:
                ch = self.basic_tokenizer._run_strip_accents(ch)
                ch = ch.lower()
            ch = self.basic_tokenizer._clean_text(ch)
            tmp += ch
            char_mapping.extend([i] * len(ch))

        text, token_mapping, offset = tmp, [], 0
        for token in tokens:
            if (token[0] == '[') and (token[-1] == ']') and bool(token):
                token_mapping.append([])
            else:
                if token[:2] == '##':
                    token = token[2:]
                start = text[offset:].index(token) + offset
                end = start + len(token)
                token_mapping.append(char_mapping[start:end])
                
                offset = end
        return token_mapping

    def tokenize(self, text):
        split_tokens = []
        for token in self.basic_tokenizer.tokenize(text):
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)
        return split_tokens
    def id_to_token(self, index):
        return self.id2token.get(index, self._token_unk)

    def token_to_id(self, token):
        return self.token2id.get(token, self.token2id[self._token_unk])

    def build_inputs_with_special_tokens(self, first_token_id, second_token_id=None):

        start = [self.token_to_id(self._token_start)]
        end = [self.token_to_id(self._token_end)]
        if second_token_id is not None:
            return start +  first_token_id + end + second_token_id + end
        return start + first_token_id + end
    
    
    def create_token_type_ids_from_sequences(self, first_token_id, second_token_id=None):
        start = [self.token_to_id(self._token_start)]
        end = [self.token_to_id(self._token_end)]
        if second_token_id is not None:
            return len(start + first_token_id + end) * [0] + len(second_token_id + end) * [1] 
        return len(start + first_token_id + end) * [0]
    
    def create_feature(self, tokens):
        tokens = [self.token_to_id(token) for token in tokens]
        tokens_id = self.build_inputs_with_special_tokens(tokens)
        # tokens_masks = self.get_special_tokens_mask(tokens)
        tokens_masks = [1] * len(tokens_id)
        segment_id = self.create_token_type_ids_from_sequences(tokens)
        return tokens_id, tokens_masks, segment_id


class BasicTokenizer(object):
    def __init__(self, do_lower_case=True, tokenize_chinese_chars=True):
        self.do_lower_case = do_lower_case
        self.tokenize_chinese_chars = tokenize_chinese_chars

    def tokenize(self, text):
        if isinstance(text, str):
            text = self._clean_text(text)
            if self.tokenize_chinese_chars:
                text = self._tokenize_chinese_chars(text)
            orig_tokens = whitespace_tokenize(text)
        else:
            orig_tokens = text
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case:
                token = token.lower()
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token))
        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text):
        """Splits punctuation on a piece of text."""
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1
        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
                (cp >= 0x3400 and cp <= 0x4DBF) or  #
                (cp >= 0x20000 and cp <= 0x2A6DF) or  #
                (cp >= 0x2A700 and cp <= 0x2B73F) or  #
                (cp >= 0x2B740 and cp <= 0x2B81F) or  #
                (cp >= 0x2B820 and cp <= 0x2CEAF) or
                (cp >= 0xF900 and cp <= 0xFAFF) or  #
                (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
            return True

        return False

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)


class WordpieceTokenizer(object):
    """Runs WordPiece tokenization."""
    def __init__(self, vocab, unk_token="[UNK]", max_inputs_chars_per_word=200):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_inputs_chars_per_word = max_inputs_chars_per_word

    def tokenize(self, text):
        """Tokenizes a piece of text into its word pieces.

        This uses a greedy longest-match-first algorithm to perform tokenization
        using the given vocabulary.

        For example:
          input = "unaffable"
          output = ["un", "##aff", "##able"]

        Args:
          text: A single token or whitespace separated tokens. This should have
            already been passed through `BasicTokenizer`.

        Returns:
          A list of wordpiece tokens.
        """
        output_tokens = []
        text = whitespace_tokenize(text) if isinstance(text, str) else text
        for token in text:
            chars = list(token)
            if len(chars) > self.max_inputs_chars_per_word:
                output_tokens.append(self.unk_token)
                continue
            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end
            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens

def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
            (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False
