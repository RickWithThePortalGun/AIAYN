import json

notebook_content = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Machine Translation: Attention Is All You Need\n",
                "\n",
                "Complete pipeline for machine translation using RNN and Transformer models."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import sys\n",
                "import torch\n",
                "import torch.nn as nn\n",
                "from tqdm import tqdm\n",
                "import matplotlib.pyplot as plt\n",
                "import pandas as pd\n",
                "import warnings\n",
                "warnings.filterwarnings('ignore')\n",
                "\n",
                "sys.path.insert(0, 'src')\n",
                "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
                "print(f'Using device: {device}')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Run this cell to execute the full pipeline\n", "\n", "The script below will handle data preparation, training, and evaluation."]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Execute the data preparation script\n",
                "exec(open('src/data/prepare_data.py').read())"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.9.6"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open('Machine_Translation_Pipeline.ipynb', 'w') as f:
    json.dump(notebook_content, f, indent=1)

print("Notebook created successfully!")

