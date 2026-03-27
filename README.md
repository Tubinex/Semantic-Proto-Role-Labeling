## Semantic-Proto-Role-Labeling with NLI

### Concept
This project explores the task of SPRL through the lens of NLI.
The idea behind this unconventional approach is to remodel Dowty-style properties into fitted hypotheses and predict their entailment by the target texts as premises.

### Installation

Clone the repo and install dependencies:

on linux/ macOS:
```bash
git clone https://github.com/Tubinex/Semantic-Proto-Role-Labeling
cd Semantic-Proto-Role-Labeling
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
on Windows (cmd):
```bash
git clone https://github.com/Tubinex/Semantic-Proto-Role-Labeling
cd Semantic-Proto-Role-Labeling
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
This will create a virtual environment inside your project folder and install the needed libraries and dependencies
there instead of globally, which is the recommended work flow.

The project was tested on python versions 3.11 and 3.12.

### Explore

The project uses interactive jupyter notebooks to guide you through the functionality of the modules in context.

Many modern IDEs are able to handle notebooks either natively or via extensions or plugins and should prompt you to download them when faced with one.
Alternatively, you can enter your virtual environment and run jupyter in your browser:

Assuming you are still in the same folder in your terminal:
```bash
jupyter notebook
```
A new tab should open in your browser. Leave the terminal running.
### Model and Data

We use the pretrained model [roberta-large-mnli](https://huggingface.co/FacebookAI/roberta-large-mnli) for probing and then fine-tune it on the spr1 dataset introduced by Reisinger et al.

### Findings

This work demonstrates that recasting SPRL as NLI is a principled and effective approach that achieves results
competitive with dedicated neural models while offering greater interpretability through its explicit hypothesis
structure. Refer to the [report](./Semantic_Proto_Role_Labeling_with_NLI.pdf) for details.

### Context

This group project was made in the context of the "Formal Semantics" lecture at Heidelberg University in the semester WS25/26.

### Authors

Marc Hauck, Polina Degtyarenko, Antonio Maria dos Santos Coelho, Nathanael Meyer  
Group "die Semantiger"

### MIT License
See `LICENSE` file for details.