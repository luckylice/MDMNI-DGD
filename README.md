# MDMNI-DGD

## What is it?

MDMNI-DGD is a graph neural network approach for druggable gene discovery based on the integration of multi-omics data and multi-view network. The input of MDMNI-DGD consists of the feature matrix of genes and gene association networks, and the output is a score representing the druggability of each gene. 

## Requirements

python > 3.6.0

torch = 1.9.0

torch-geometric = 2.0.3

numpy = 1.19.2 

pandas = 1.1.5

networks = 2.5.1

scipy = 1.5.4 

matplotlib = 3.3.4 

scikit-learn = 0.24.2 

## Usage

The whole workflow of MDMNI-DGD is comprised of three steps: 

- Feature matrix generation and multi-view network construction. 
- Feature extraction and fusion. 
- Druggability scores prediction.  

### Command Line Tool

We can run MDMNI-DGD in the following way.

```python
# To modify the input feature file, you can make adjustments to the "train_name" parameter. Similarly, by modifying the "networks" parameter, you can select different gene correlation networks.
python main.py --train_name Feature_data --networks 1 2 3 4 5 6 --k 10
# During the execution of the code, if the network files are not found, the code will automatically execute "network.py" to generate them.
```

After training MDMNI-DGD, executing the best model on all 20,544 genes to obtain potentially druggable genes. The specific implementation code is as follows.

```python
# Input the feature file for all genes along with the corresponding multi-view network
python case_study.py --train_name All_genes --networks 1 2 3 4 5 6
```

### Data Format

- In the feature matrix, rows represent genes, and columns represent multi-omics features. The number of samples in each multi-perspective network should be consistent with the number of samples in the feature matrix.
- During the training process of MDMNI-DGD, sample labels are required. The labels of the samples should be placed in the last column of the feature matrix.
