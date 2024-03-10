# Optimizing LLMs Decoding Strategies for News Generation

This repository contains the code and report developed when doing the Praktikum **Machine Learning for Natural Language Processing Applications** at the Technical University of Munich.

There are two parts to our work:
- Fine-tuning models and saving the generated texts with different decoding strategies: `ft_gen`
- Evaluating the obtained neural news articles according to different automated metrics: `eval`

## Part 1: Fine-tuning models and generating texts with different decoding strategies

This section contains 5 Jupyter Notebook files, each one with respect to a different model (GPT-2, T5, GPT-3.5, Bert2Bert, Llama 2), meant to be run on Google Colab to allow for GPU use. As such, to recreate our results simply open each file on Google Colab and run the texts. Make sure to have the appropriate directories set up on your Google Drive for the texts to be saved.


## Part 2: Evaluating the obtained texts with automated metrics

This section contains two Python scripts, `evaluation.py` and `plotting.py`. 

The script `evaluation.py` will run automated metrics for a certain model's generated texts. Make sure to change the variable `PATH` of where the texts are stored and the model name of where to store the results. The current version is set to GPT-3.5.
Note that there is some setup required to run this file:
- Required packages (available via `pip`): `transformers`, `scikit-learn`, `py-readability-metrics`, `tqdm`, `rouge`
- `python -m nltk.downloader punkt` 
- For BLEURT, follow these steps (taken from https://github.com/google-research/bleurt):
```
# Download and install the model
pip install --upgrade pip  # ensures that pip is current
git clone https://github.com/google-research/bleurt.git
cd bleurt
pip install .

# Download the BLEURT-base checkpoint (pretrained model).
wget https://storage.googleapis.com/bleurt-oss-21/BLEURT-20.zip .
unzip BLEURT-20.zip
python -m bleurt.score_files -candidate_file=/path/to/file -reference_file=/path/to/file bleurt_checkpoint=BLEURT-20 > /path/to/file
```
The latter command will compute a score for each line in the candidate file and reference file and save it to an output file. Make sure to run this for each decoding strategy before running the `evaluation.py` script and to rewrite the file with the path you saved results to.

The script `plotting.py` will plot the automated metrics for easier visualization for one model. Make sure to update the path of where each results file is stored.
There is also some setup required:
- Required packages (available via `pip`): `pandas`, `plotly`
