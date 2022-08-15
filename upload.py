import transformers
from transformers import T5ForConditionalGeneration, T5Tokenizer, TFT5ForConditionalGeneration

model = T5ForConditionalGeneration.from_pretrained("/home/ubuntu/20220812/models/t5_dialog_jp")
model.push_to_hub("t5_dialog_jp", use_temp_dir=True)

tokenizer = T5Tokenizer.from_pretrained("/home/ubuntu/20220812/models/t5_dialog_jp")
tokenizer.push_to_hub("t5_dialog_jp", use_temp_dir=True)

# tf_model = TFT5ForConditionalGeneration.from_pretrained("/home/ubuntu/20220812/models/t5_dialog_jp", from_pt=True)
# tf_model.push_to_hub("t5_dialog_jp")
