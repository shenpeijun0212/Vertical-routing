# -*- coding: utf-8 -*-
"""
Standalone Example for WMT Dataset Evaluation using the DRT Template.

This script demonstrates the logic of the "Deep Reasoning Translation" (DRT)
domain-specific template, as described in the "Vertical Routing" paper,
applied to translation tasks like those in the WMT dataset. It is a
self-contained example using mock objects to simulate the workflow.

Workflow for Deep Reasoning Translation:
1.  **Keyword Extraction Stage (Large Model)**: A mock "large model" extracts
    key terms from the source sentence and provides their translations. This
    is the high-importance stage that sets the translation's semantic foundation.
2.  **Sentence Synthesis Stage (Small Model)**: A mock "small model" takes
    the original sentence and the extracted keywords to synthesize the final,
    fluent translation, ensuring consistency with the critical keywords.
"""
import os
import re
import time
from typing import Dict, Any, Tuple, List

# --- Mock Objects for Demonstration ---
# In a real scenario, these would be imported from the transformers library
class MockPreTrainedModel:
    """A mock class to simulate a Hugging Face model."""
    def __init__(self, model_type="large"):
        self.model_type = model_type

    def generate(self, input_ids, **kwargs) -> List[List[int]]:
        # Simulate model generation based on the prompt type (keywords vs translation)
        if "Keywords:" in MOCK_TOKENIZER.decode(input_ids[0]):
            # This is a keyword extraction prompt for the large model
            return [[0] * 20] # Return a fixed-length list of mock token IDs for the first part
        else:
            # This is a sentence translation prompt for the small model
            return [[1] * 50] # Return a different list of mock token IDs for the second part
            
    def to(self, device):
        # Mock device placement
        return self

class MockTokenizer:
    """A mock class to simulate a Hugging Face tokenizer."""
    def __call__(self, text_list, return_tensors="pt"):
        # Simulate tokenization
        return {"input_ids": [[i for i, char in enumerate(text_list[0])]]}

    def encode(self, text, add_special_tokens=False):
        # Simulate encoding
        return [123]

    def decode(self, token_ids, skip_special_tokens=True):
        # Simulate decoding based on the mock token IDs from the model
        if token_ids and token_ids[0] == 0:
            return "'Vertical Routing' as '垂直路由', 'framework' as '框架', 'demonstrates' as '展示了', 'collaboration' as '协作'"
        elif token_ids and token_ids[0] == 1:
            return "这个垂直路由框架展示了大型和小型模型之间的协作。"
        return "".join([str(t) for t in token_ids])

# Instantiate the mock tokenizer globally
MOCK_TOKENIZER = MockTokenizer()

# --- 1. Configuration Module ---
class Config:
    """Minimal configuration for the template."""
    DEVICE: str = "cpu"
    # The maximum number of tokens for the large model to generate when extracting keywords
    KEYWORDS_MAX_NEW_TOKENS: int = 100
    # The maximum number of tokens for the small model to generate when translating the sentence
    TRANSLATE_MAX_NEW_TOKENS: int = 200
    TEMPERATURE: float = 0.1
    TOP_P: float = 0.9
    DO_SAMPLE: bool = False


# --- 2. Model Management Module ---
class MockModelManager:
    """A mock model manager that provides mock models and tokenizers."""
    def __init__(self, config: Config):
        self.config = config
        print("Loading MOCK models and tokenizers...")
        self.large_model = MockPreTrainedModel(model_type="large")
        self.small_model = MockPreTrainedModel(model_type="small")
        self.large_tokenizer = MOCK_TOKENIZER
        self.small_tokenizer = MOCK_TOKENIZER
        print("✅ Mock models and tokenizers are ready.")


# --- 3. Template Definition Module ---
class BaseTemplate:
    """Base class for all routing templates."""
    def __init__(self, model_manager: MockModelManager, config: Config):
        self.mm = model_manager
        self.config = config
        self.name = self.__class__.__name__

    def _generate(self, model: MockPreTrainedModel, tokenizer: MockTokenizer, prompt: str, max_new_tokens: int) -> Tuple[str, int]:
        """Generic text generation function using mock objects."""
        inputs = tokenizer([prompt], return_tensors="pt")
        
        # The mock decoder will handle the stopping logic
        generated_ids = model.generate(inputs['input_ids'])
        response = tokenizer.decode(generated_ids[0])

        # A simple token count approximation
        num_tokens = len(response.split())
        return response, num_tokens

    def run(self, query: str) -> Tuple[str, int, int]:
        """This method should be implemented by subclasses."""
        raise NotImplementedError

class DeepReasoningTranslationTemplate(BaseTemplate):
    """
    Implements the Deep Reasoning Translation (DRT) domain-specific template.
    """
    def run(self, query: str) -> Tuple[str, int, int]:
        """Executes the DRT workflow."""
        print("--- Executing Deep Reasoning Translation (DRT) Template ---")
        
        # --- Stage 1: Keyword Extraction (Large Model) ---
        print("\n[Stage 1] Extracting keywords with Large Model...")
        keyword_prompt = (
            "You are a professional translation expert. Please strictly follow the two steps below to translate the English sentence I provide:\n"
            "Step 1: Extract Keywords. Extract the keywords and list them in the following format: 'English word' as 'Chinese translation'.\n"
            "Step 2: Translate the Sentence. Based on the keywords extracted in Step 1, translate the entire English sentence into Chinese.\n\n"
            f"English sentence: {query}\n"
            "Keywords:"
        )
        keywords, large_model_tokens = self._generate(
            model=self.mm.large_model,
            tokenizer=self.mm.large_tokenizer,
            prompt=keyword_prompt,
            max_new_tokens=self.config.KEYWORDS_MAX_NEW_TOKENS
        )
        print(f"Extracted Keywords: '{keywords}'")

        # --- Stage 2: Sentence Translation (Small Model) ---
        print("\n[Stage 2] Translating sentence with Small Model...")
        translation_prompt = (
            f"English sentence: {query}\n"
            f"Keywords: {keywords}\n"
            "Translation:"
        )
        translation, small_model_tokens = self._generate(
            model=self.mm.small_model,
            tokenizer=self.mm.small_tokenizer,
            prompt=translation_prompt,
            max_new_tokens=self.config.TRANSLATE_MAX_NEW_TOKENS
        )
        print(f"Generated Translation: '{translation}'")

        total_response = f"--- Keywords (Generated by Large Model) ---\n{keywords}\n\n--- Final Translation (Generated by Small Model) ---\n{translation}"
        return total_response, large_model_tokens, small_model_tokens


# --- 4. Main Execution Function ---
def main():
    """Main function to demonstrate the DRT template."""
    print("🚀 Initializing standalone DRT demo...")
    
    # Configuration and mock manager setup
    config = Config()
    mock_manager = MockModelManager(config)
    
    # Instantiate the template
    drt_template = DeepReasoningTranslationTemplate(mock_manager, config)
    
    # Define a WMT-style sample query
    sample_query = "The Vertical Routing framework demonstrates collaboration between large and small models."
    print(f"\nSample English Sentence: \"{sample_query}\"")
    
    # Run the template
    final_response, keyword_tokens, translation_tokens = drt_template.run(sample_query)
    
    # Print the final results
    print("\n--- ✅ Final Combined Output ---")
    print(final_response)
    print("\n--- 📊 Summary ---")
    print(f"Tokens from Large Model (Keywords): {keyword_tokens}")
    print(f"Tokens from Small Model (Translation): {translation_tokens}")
    print(f"Total Generated Tokens: {keyword_tokens + translation_tokens}")
    print("\nNote: In a real WMT evaluation, the generated translation would be compared against a reference to calculate metrics like BLEU.")

if __name__ == "__main__":
    main()
