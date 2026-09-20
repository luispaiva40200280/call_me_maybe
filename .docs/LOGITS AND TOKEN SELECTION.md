
> [!note]
> _auto-regressive_ language generation is based on the assumption that the probability distribution of a word sequence can be decomposed into the product of conditional next word distributions: -- The chance of generating a specific sentence is calculated by guessing one word at a time, based on the prompt and all the words guessed so far, and multiplying those chances together. 

$$
P(w1:T∣W0)=∏t=1TP(wt∣w1:t−1,W0) ,with w1:0=∅, P(w_{1:T} | W_0 ) = \prod_{t=1}^T P(w_{t} | w_{1: t-1}, W_0) \text{ ,with } w_{1: 0} = \emptyset, P(w1:T​∣W0​)=t=1∏T​P(wt​∣w1:t−1​,W0​) ,with w1:0​=∅,
$$
### LOGITS: __Function of probabilities__

> *Logits are the raw, unnormalized voting scores a neural network assigns to every possible next token. The model then uses a formula to turn those logits into probabilities to make its final guess*. Basically **mathematical compatibility measurement**. It represents how closely a specific token aligns with what the neural network is currently *"thinking"* about your prompt.

##### 1. The Context Vector (What the model is "thinking")
- As the LLM reads a prompt (e.g., `"What is the sum of"`), it passes the tokens of the prompt through its many layers. By the time it reaches the very end of the network, the model has generated a **Hidden State** — a massive array of numbers (often thousands of numbers long) that represents a complex, mathematical summary of the sentence's context and what concepts should logically follow. _This array is, in a way a mathematical shape representing the "blank space" waiting to be filled._
##### 2. The Token Vectors (The dictionary's definitions)
- Simultaneously, the model has a massive matrix containing the mathematical definition for every single one of its 151,000 tokens. Every token (like `" 40"`, `" apple"`, or `"}"`) is represented by its own array of numbers of the exact same length as the Hidden State.
##### 3. The Math (The Dot Product)
- To figure out which token belongs in the blank space, the model performs a mathematical operation called a **[[Dot Product]]**. It takes the Context Vector and multiplies it against every single Token Vector in the dictionary.
	- If the numbers in the Context Vector line up perfectly with the numbers in a Token Vector, the math produces a **large positive number** (e.g., `18.5`).
	- If the numbers conflict or point in completely opposite conceptual directions, the math produces a **negative number** (e.g., `-5.2

**That resulting number is the logit.**

### 2.The Probability Formula (Softmax):

Because working with random numbers like `14.5` and `-3.2` is messy, the model passes all 151,000 logits through a mathematical formula called **Softmax**. Softmax acts like a compressor. It squishes all those raw logits down so they fit exactly between 0 and 1, and forces them all to add up to 100%.

- `" sum"` = `0.95` (95% chance) 
- `" apple"` = `0.01` (1% chance)
- `" zebra"` = `0.04` (4% chance)