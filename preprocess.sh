# tokenize 
python tokenize.py

echo "Tokenize done."

# bpe
subword-nmt learn-bpe -s 32000 < data/NEU.en.tok > codes_file.bpe
subword-nmt apply-bpe -c codes_file.bpe < data/NEU.en.tok > data/NEU.en.tok.bpe

echo "All done."