# [[Dot Product]]

> [!Note]
> A **dot product** is a mathematical operation that compares two arrays of numbers (called vectors) to measure how similar they are or how much they "point in the same direction."
### How the Math Works

- To calculate a dot product, you multiply the corresponding numbers from each array together, and then add all those results into a single final number, **Simple Example:** _If Vector A is $[2, 3]$ and Vector B is $[4, 1]$:_ $$(2 \times 4) + (3 \times 1) = 8 + 3 = 11$$The dot product is **11**.

### What the Result Means:

When comparing two vectors of data, the final number tells you their relationship:

- **Large Positive Number:** The vectors are highly similar (they align conceptually).
- **Zero:** The vectors have nothing to do with each other (they are completely unrelated).
- **Large Negative Number:** The vectors are exact opposites (they point in conflicting directions).

### In the Context of LLMs (Logits)

In a Large Language Model, the dot product is the exact mathematical operation used to calculate the **logit**.
The model calculates the dot product between the **Context Vector** (a massive array of numbers summarizing the sentence so far) and a **Token Vector** (the array of numbers defining a specific word in the dictionary).

- If the token conceptually fits the context, the dot product yields a high positive number (a high logit).  
- If the token makes no sense in that context, the dot product yields a negative number (a low logit).