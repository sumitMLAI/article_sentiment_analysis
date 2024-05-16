
from transformers import BartTokenizer, BartForConditionalGeneration


def summarize_text(text):
    # Load pre-trained tokenizer and model
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

    # Tokenize input text
    inputs = tokenizer([text], max_length=1024, return_tensors='pt', truncation=True)

    # Generate summary
    summary_ids = model.generate(inputs['input_ids'], max_length=80, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary
