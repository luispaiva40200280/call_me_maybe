from llm_sdk.llm_sdk import Small_LLM_Model


def main():
    print("Loading the model and tokenizer...")
    # Initialize the model (defaults to Qwen/Qwen3-0.6B)
    model = Small_LLM_Model()

    # The phrase we want to test
    phrase = "What is the sum of 40 and 2?"
    print(f"\nOriginal phrase: '{phrase}'")

    # 1. ENCODE: Convert the phrase into Token IDs
    # The encode method returns a 2-D tensor (e.g., [[id1, id2, ...]])
    input_ids_tensor = model.encode(phrase)

    # Convert the tensor to a standard Python list for the first batch
    token_ids = input_ids_tensor[0].tolist()

    print(f"\nTotal tokens: {len(token_ids)}")
    print(f"Token IDs array: {token_ids}\n")

    # 2. DECODE: Translate each individual ID back into text
    print("Token Breakdown:")
    print("-" * 30)
    for token_id in token_ids:
        # The decode method expects a list or tensor, so we wrap the
        # single ID in a list
        token_string = model.decode([token_id])

        # We use repr() to clearly see spaces (it will print ' word'
        # instead of just  word)
        print(f"ID {token_id:<6} -> {repr(token_string)}")


if __name__ == "__main__":
    main()
