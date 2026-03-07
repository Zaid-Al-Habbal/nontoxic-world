# Preprocessing Plan

## 1. Text Normalization Rules
- Convert all text to lowercase to ensure uniformity.
- Remove extra whitespace to clean the text.
- Remove user names.
- Replace URLs with a placeholder token [URL].
- Remove wiki markup, converting [[link|text]] to just "text".


## 2. Tokenization
- I used pretrained BERT tokenizer (bert-base-uncased)
- BertTokenizer(name_or_path='bert-base-uncased', vocab_size=30522, model_max_length=512, padding_side='right', truncation_side='right', special_tokens={'unk_token': '[UNK]', 'sep_token': '[SEP]', 'pad_token': '[PAD]', 'cls_token': '[CLS]', 'mask_token': '[MASK]'}, added_tokens_decoder={
	0: AddedToken("[PAD]", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	100: AddedToken("[UNK]", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	101: AddedToken("[CLS]", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	102: AddedToken("[SEP]", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	103: AddedToken("[MASK]", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
})

- We set `max_length` to 256 based on the comment length distribution, which covers the majority of comments while avoiding extreme outliers.

- We Added a special token for URLs: [URL]

- preproc_train contains the preprocessed training data without tokenization