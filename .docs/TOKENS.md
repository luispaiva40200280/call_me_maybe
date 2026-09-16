## What Are Tokens?

>At its core, a **token** is the fundamental unit of data processed by a Large Language Model (LLM). Because neural networks only understand numbers, text must be translated into a mathematical format before the model can process it.

**==Tokenization==** is the process of breaking down human-readable text into these numerical values. It operates on three main components:

### 1. The Sequence (Subword Units)
> Instead of rigidly chopping text letter-by-letter or word-by-word, modern LLMs chop text into optimized pieces called **subwords**.

- These sequences can be single characters, partial words, whole words, or even words attached to punctuation  
- Crucially, tokenizers preserve exact formatting, such as leading spaces, which are often represented by special characters like `Ġ`
- This chopping is done using specialized algorithms like Byte Pair Encoding (BPE) or SentencePiece
### 2. The Dictionary (The Vocabulary)
> The model possesses a massive, pre-saved mapping—known as the **vocabulary file**—created during its initial training phase.

- This dictionary stores every single subword sequence the model is capable of recognizing
- In this project, this dictionary is accessed using the `get_path_to_vocab_file()` method provided by the LLM_SDK
### 3. The Value (Input IDs)
> The **Token ID** (or Input ID) is simply the integer index number indicating exactly where a specific text sequence lives inside the dictionary.

- When we **encode** text, we look up sequences in the dictionary and return their Token IDs
- When we **decode** text, we look up Token IDs and return their corresponding text sequences.
### 📝 Concrete Example

If we feed the sentence `"What is the sum"` into the tokenizer, the process looks like this:

1. **Text Parsing**: The tokenizer splits the string into subword sequences, preserving the spaces as `Ġ`.    
    - _Result:_ `["What", "Ġis", "Ġthe", "Ġsum"]`
2. **Dictionary Lookup**: The tokenizer finds these exact strings in the model's vocabulary file.  
3. **ID Assignment**: The tokenizer returns the numerical index for each sequence
    - _Result (Input IDs):_ `[892, 318, 262, 4771]`

>[!note]
>_The numbers above are illustrative; exact IDs depend entirely on the specific model's dictionary._

