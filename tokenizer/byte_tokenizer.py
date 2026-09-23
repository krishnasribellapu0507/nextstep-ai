class ByteTokenizer:
    """
    NextStep's first tokenizer.

    Every UTF-8 byte becomes one token from 0-255.
    No HuggingFace tokenizer.
    No Gemini tokenizer.
    No external tokenizer model.
    """

    vocab_size = 256

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, tokens: list[int]) -> str:
        data = bytes(int(t) % 256 for t in tokens)
        return data.decode("utf-8", errors="replace")


if __name__ == "__main__":
    tokenizer = ByteTokenizer()

    text = "NextStep teaches matrices."
    tokens = tokenizer.encode(text)

    print("Original:", text)
    print("Tokens:", tokens)
    print("Decoded:", tokenizer.decode(tokens))
