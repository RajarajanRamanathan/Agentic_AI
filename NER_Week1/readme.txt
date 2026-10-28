Models used:

Summarization:	sshleifer/distilbart-cnn-12-6
Sentiment analysis:	distilbert/distilbert-base-uncased-finetuned-sst-2-english
Named entity recognition:	dslim/bert-base-NER

Commands:
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned 
venv\Scripts\Activate.ps1 
python -m pip install --upgrade pip
python -m pip install "transformers==4.57.6" "torch"  
