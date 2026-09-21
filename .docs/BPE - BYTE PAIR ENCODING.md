>[!info]
In [computing](https://en.wikipedia.org/wiki/Computing "Computing"), **byte-pair encoding** (**BPE**),[[1]](https://en.wikipedia.org/wiki/Byte-pair_encoding#cite_note-CUsersJ_Gage_1994-1)[[2]](https://en.wikipedia.org/wiki/Byte-pair_encoding#cite_note-2) or **diagram coding**,[[3]](https://en.wikipedia.org/wiki/Byte-pair_encoding#cite_note-3) is an [algorithm](https://en.wikipedia.org/wiki/Algorithm "Algorithm"), first described in 1994 by Philip Gage, for encoding strings of text into smaller strings by creating and using a translation table.[[4]](https://en.wikipedia.org/wiki/Byte-pair_encoding#cite_note-4) A slightly modified version of the algorithm is used in [large language model](https://en.wikipedia.org/wiki/Large_language_model "Large language model") tokenizers.
### BPE - BYTE PAIR ENCODING Algorithm

- The original version of the algorithm focused on compression. It replaces the highest-frequency pair of [bytes](https://en.wikipedia.org/wiki/Byte "Byte") with a new byte that was not contained in the initial dataset. A *[^1]lookup table* of the replacements is required to rebuild the initial dataset. The modified version builds "tokens" (units of recognition) that match varying amounts of source text, from single characters (including single digits or single punctuation marks) to whole words (even long compound words)
#### How does it operate ?
 The original BPE algorithm operates by iteratively replacing the most common contiguous sequences of characters in a target text with unused 'placeholder' bytes. The iteration ends when no sequences can be found, leaving the target text effectively compressed. Decompression can be  performed by reversing this process, querying known placeholder terms against their corresponding denoted sequence, using a lookup table. In the original paper, this lookup table is encoded and stored alongside the compressed text.
- __*Exemple:*__ 
	- Suppose the data to be encoded is:
		-> ` aaabdaaabac `

	- The byte pair "aa" occurs most often, so it will be replaced by a byte that is not used in the data, such as "Z". Now there is the following data and replacement table:
		-> ZabdZabac
		-> Z=aa
	- Then the process is repeated with byte pair "ab", replacing it with "Y":
		-> ZYdZYac
		-> Y=ab
		-> Z=aa

	- The only literal byte pair left occurs only once, and the encoding might stop here. Alternatively, the process could continue with [recursive](https://en.wikipedia.org/wiki/Recursion "Recursion") byte-pair encoding, replacing "ZY" with "X":
		-> XdXac
		-> X=ZY
		-> Y=ab
		-> Z=aa

[^1]: lookup table ([LUT](https://en.wikipedia.org/wiki/Lookup_table)):
> [!info]

> - **What it is:** A LUT is an array that acts like a built-in shortcut. Instead of forcing a computer to calculate a math problem from scratch every time it needs an answer, it uses a simpler array indexing operation (called direct addressing) to pull a pre-calculated answer.
> - **Why it is useful:** It can save a massive amount of processing time. Grabbing a value straight from computer memory is usually much faster than running a heavy, "expensive" computation or input/output operation.
> - **How they are made:** These tables can be precalculated and saved ahead of time, created when a program starts up (known as memoization), or even built right into physical hardware like FPGAs.
> - **Other uses:** LUT's are also used to quickly check if an input is valid by matching it against a list of approved or unapproved items.
> - **LUT's vs. Hash Tables:** Unlike a hash table—which uses a special formula (a hash function) to figure out which slot a value is stored in—a LUT stores the value directly in the slot matching its key, making it directly addressable.

