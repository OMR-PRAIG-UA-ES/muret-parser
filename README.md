<p align='center'>
  <a href='https://praig.ua.es/'><img src='https://i.imgur.com/Iu7CvC1.png' alt='PRAIG-logo' width='100'></a>
</p>

<h1 align='center'>Python parser for MuRET-format datasets</h1>


<p align='center'>
  <img src='https://img.shields.io/badge/python-3.10.0-orange' alt='Python'>
</p>


## Usage

To use the Python parser for MuRET-format datasets, follow these steps:

1. **Install the required packages:**

```python
pip install -r requirements
```

2. **Run the parser script:**

```python
python main.py --muret_json_folder_path <path_to_muret_json> --output_folder_path <output_path> [--k <num_folds>]
```
- Replace `<path_to_muret_json>` with the actual path to the folder containing your MuRET-format dataset JSON files.
- Replace `<output_path>` with the desired path for the output folder where the script will generate the processed data.
- The `--k <num_folds>` argument is optional. Use it to specify the number of folds for k-fold cross-validation (default is 5 folds).
