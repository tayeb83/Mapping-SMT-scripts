# Running the Lexical Mapping Script
This guide will help you set up and run the lexical mapping script, which compares two CSV files based on their "code" and "label" columns and outputs the results, including all additional columns, into an Excel file.

## Prerequisites
- Python 3.x installed on your system.
- Access to a terminal or command prompt.

##I nstallation
1- Install Required Libraries

Open your terminal and install the necessary Python packages:

```

pip install pandas nltk string-grouper

```

```
import nltk
nltk.download('stopwords')

```
2- Prepare Your Files

- Save the provided Python script as lexical_mapping.py.
- Ensure your CSV files are ready. They must contain at least "code" and "label" columns. Additional columns are allowed and will be included in the output.
- (Optional) If using additional French stopwords, ensure the file data/stopWords/additional_stop_words exists with your extra stopwords listed.

## Usage
Run the script from your terminal, specifying the paths to your input files, the output file, and the language option:

```
python lexical_mapping.py --input1 path/to/terminology1.csv --input2 path/to/terminology2.csv --output path/to/output.xlsx --language auto
```
- Replace `path/to/terminology1.csv` and `path/to/terminology2.csv` with the paths to your input CSV files. 
- Replace `path/to/output.xlsx` with the path where you want to save the output Excel file.

The `--language` option specifies the language for preprocessing. It can be set to french, english, or auto. If set to auto, the script defaults to English unless specifically coded otherwise.

### Notes
- Customizing the Script: Depending on your data structure or requirements, you might need to adjust the script. It assumes a simple matching logic and the presence of an additional stopwords file named additional_stop_words located at `data/stopWords/`.
- Language Support: The script includes support for French and English based on specified stopwords and stemmers. To extend support for more languages, adjust the preprocess_text function with appropriate resources.

