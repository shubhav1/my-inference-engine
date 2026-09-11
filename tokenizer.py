
import regex
import json

# bytes to unicode
def bytes_to_unicode():
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))

# tokenizer
class QwenTokenizer:
    # pre-tokenizer regex used by Qwen's tokenizer.json
    PAT = regex.compile(r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+""")

    def __init__(self, model_path):
        with open(f"{model_path}/vocab.json", encoding="utf-8") as f:
            self.encoder = json.load(f)
        self.decoder = {i: t for t, i in self.encoder.items()}

        with open(f"{model_path}/merges.txt", encoding="utf-8") as f:
            merges = [tuple(line.split()) for line in f.read().splitlines()[1:] if line]
        self.bpe_ranks = {pair: rank for rank, pair in enumerate(merges)}

        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.cache = {}

    @staticmethod
    def get_pairs(word):
        return {(word[i], word[i + 1]) for i in range(len(word) - 1)}

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]

        word = tuple(token)
        pairs = self.get_pairs(word)
        if not pairs:
            return token

        while True:
            pair = min(pairs, key=lambda p: self.bpe_ranks.get(p, float("inf")))
            if pair not in self.bpe_ranks:
                break
            first, second = pair
            new_word = []
            i = 0
            while i < len(word):
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = self.get_pairs(word)

        result = " ".join(word)
        self.cache[token] = result
        return result

    def encode(self, text):
        ids = []
        for token in regex.findall(self.PAT, text):
            token_bytes = token.encode("utf-8")
            token_str = "".join(self.byte_encoder[b] for b in token_bytes)
            for bpe_token in self.bpe(token_str).split(" "):
                ids.append(self.encoder[bpe_token])
        return ids

    def decode(self, ids):
        text = "".join(self.decoder[i] for i in ids)
        return bytearray(self.byte_decoder[c] for c in text).decode("utf-8", errors="replace")
