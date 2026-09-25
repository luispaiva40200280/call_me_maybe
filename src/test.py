import json
from transformers import AutoTokenizer

# Adjust these imports to match your project structure
from src.schemas import FunctionDef
from src.fsm import StateMachine as FSM
from src.fsm import JSONState


def run_isolated_fsm_test(json_path: str):
    # 1. LOAD YOUR REAL JSON FILE
    # Replace with the actual path to your JSON definitions
    # json_path = "functions_definition.json" 
    try:
        with open(json_path, "r") as f:
            raw_data = json.load(f)
            # Assuming your JSON is a list of function dictionaries
            schemas = [FunctionDef(**schema) for schema in raw_data]
    except FileNotFoundError:
        print(f"Error: Could not find {json_path}")
        return

    # Initialize your FSM with the real schemas
    fsm = FSM(schemas=schemas)

    # 2. LOAD THE REAL LLM TOKENIZER
    # Replace with the exact model you intend to use (e.g., "meta-llama/Meta-Llama-3-8B")
    model_name = "gpt2" 
    print(f"Loading Tokenizer for: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 3. DEFINE A PERFECT TARGET JSON
    # IMPORTANT: Change this string so it accurately represents a valid 
    # function call based on your specific functions_definition.json!
    target_json = '{"name": "your_actual_function_name", "parameters": {"your_actual_param": 42}}'

    # 4. CHOP IT INTO LLM TOKENS
    token_ids = tokenizer.encode(target_json)
    print(f"\n--- STARTING FSM TEST ({len(token_ids)} tokens) ---\n")

    # 5. FEED TOKENS TO THE FSM
    for token_id in token_ids:
        # Decode the single token exactly how the LLM generates it
        token_text = tokenizer.decode([token_id])
        
        print(f"CURRENT STATE : {fsm.state.name}")
        # We use repr() to clearly see spaces and newlines (e.g., ' "name"' vs '"name"')
        print(f"LLM GENERATED : {repr(token_text)}")
        
        # Check what the Enforcer allows right now
        allowed = fsm.get_allowed_strings()
        display_allowed = allowed[:8] + ["..."] if len(allowed) > 8 else allowed
        print(f"FSM ALLOWS    : {display_allowed}")
        
        # Step the FSM forward
        try:
            fsm.update_state(token_text)
        except ValueError as e:
            print(f"\n[!] FSM REJECTED THE TOKEN: {e}")
            break
            
        print("-" * 50)

    print(f"\nFINAL STATE: {fsm.state.name}")
    if fsm.state == JSONState.DONE:
        print("SUCCESS: The FSM successfully parsed the entire tokenized JSON!")
    else:
        print("INCOMPLETE: The FSM did not reach the DONE state.")

if __name__ == "__main__":
    run_isolated_fsm_test()


"""
    from src.test import run_isolated_fsm_test
    ARGS = ArgsParser()

    run_isolated_fsm_test(json_path=ARGS.functions_definition)
        lut_trie = VocabularyTrie()
        lut_trie.build_from_lut(lut)
    # import json   # use to test the loaders

        # TESTING IF THE FILES ARE OPENING AND STORING THE DATA CORRECTLY
        print("=== Prompts in json file")
        for prom in prompts:
            print(f"- {prom.values()}")

        print("\n=== Func Calling in json file ===")
        # Loop through the functions and print them cleanly
        for func in functions_calling:
            # .model_dump() (or .dict() in older Pydantic versions) converts
            # the model back to a readable dictionary
            print(json.dumps(func, indent=2))
        for index, item in enumerate(lut.items(), start=1):
            print(f"{index} -- {item}")

        # Manual verification tests
        print("Testing Trie Insertion...")
        # 1. Test root level children
        print(f"Number of distinct starting characters:
        {len(lut_trie.root.children)}")
        # 2. Test a specific known path (e.g., the '{' character)
        if "{" in lut_trie.root.children:
            brace_node = lut_trie.root.children["{"]
            print(f"Node for '{{' has token ID: {brace_node.token_id}")
            print(f"Characters that can follow
              '{{': {list(brace_node.children.keys())[:5]}") # Print first 5
        else:
            print("Error: '{' not found in Trie.")
"""
