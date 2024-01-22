# Welcome to 28934: Automation and control of yeast fermentation

### Graduate course - Technical University of Denmark
### Department of Chemical and Biochemical Engineering - Process and Systems Engineering

This .zip folder contains python tutorials for each day, following the lectures. <br>
For each day, execute the jupyter notebooks (.ipynb), they can be executed in google.colab!<br>
https://colab.research.google.com/

One way is to mount your personal google drive to your colab session to access and edit the files directly.
this can be done in colab with the following command in collab:
```
# link Google drive -- add, delete, modify files without losing all the changes
from google.colab import drive
drive.mount('/content/gdrive')
```

To run the code locally, the python environment can be loaded with conda: 
```
conda env create -f environment.yml
```
Or pip :
```
pip install -r requirements.txt
```

If the .zip folder is uploaded to your google drive, it should be visible in ***content/gdrive***.

the .py files should not be executed, they only contain functions that are called in the notebooks.

- **DAY1**: Introduction to python
- **DAY2**: Controller inplementation in python
- **DAY3**: Data generation from a fed batch model
- **DAY4**: Data smoothing and pre-treatment
- **DAY5**: Introduction to machine learning 


Enjoy! 


If you have any questions: <br>
Johan Le Nepvou De Carfort,
jlne@kt.dtu.dk<br>
PhD student at the Technical University of Denmark<br>
Department of Chemical and Bichemcal Engineering