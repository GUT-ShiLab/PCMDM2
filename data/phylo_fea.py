import pandas as pd
import numpy as np
from ete3 import NCBITaxa

# Step 1: Read the microbiome abundance table
file_path = "./IBD/Origin_data/microbiome_Train.csv"
data = pd.read_csv(file_path)
# Extract the feature-name column and abundance columns
feature_column = data.columns[0]
data[feature_column] = data[feature_column].astype(str).apply(lambda x: x.strip())

# Function to extract genus names and merge abundances
def extract_genus(feature_name):
    # Split to get the genus part and remove possible suffixes
    genus_part = feature_name.split("g__")[-1]
    genus = genus_part.split("_")[0] if "_" in genus_part else genus_part
    return genus.strip()

# Create a dictionary to store genus names and their corresponding abundances
genus_dict = {}

# Iterate over all feature names and extract genus names
for index, row in data.iterrows():
    genus = extract_genus(row[feature_column])
    abundance = row[1:].values.astype(float)  # Ensure abundance values are float

    if genus in genus_dict:
        genus_dict[genus] += abundance  # Merge abundances for the same genus
    else:
        genus_dict[genus] = abundance  # Add new genus and its abundances

# Combine genus names and abundances into a DataFrame (use original column names as new column names)
sample_names = data.columns[1:]  # Use sample names as column names
genus_names = list(genus_dict.keys())
abundance_values = np.vstack(list(genus_dict.values())).T  # Transpose to match sample/genus orientation
genus_data = pd.DataFrame(data=abundance_values, columns=genus_names, index=sample_names)

# # Save the merged data (set index=True to write sample names to the first column)
# output_path = "./data/IBD/phylogenTree_data/genus_merged.csv"
# genus_data.to_csv(output_path, index=True)
#
# print(f"Merged genus abundance table saved to: {output_path}")

# Step 2: Convert genus names to NCBI IDs
raw_name = genus_data.columns.tolist()  # Genus names
ncbi = NCBITaxa()

# Retrieve NCBI IDs for genus names
raw_id = ncbi.get_name_translator(raw_name)

# Filter out genera without NCBI IDs
valid_genus = []
valid_raw_id_list = []
for genus in raw_name:
    if genus in raw_id and len(raw_id[genus]) > 0:
        valid_genus.append(genus)
        valid_raw_id_list.append(str(raw_id[genus][0]))  # Take the first valid NCBI ID

# Remove genera without corresponding NCBI IDs (and their data)
genus_data = genus_data[valid_genus]

# **Step 2.1: Merge features that share the same NCBI ID**
merged_ncbi_dict = {}
for genus, ncbi_id in zip(valid_genus, valid_raw_id_list):
    abundance = genus_data[genus].values
    if ncbi_id in merged_ncbi_dict:
        merged_ncbi_dict[ncbi_id] += abundance
    else:
        merged_ncbi_dict[ncbi_id] = abundance

# Recreate the DataFrame with NCBI IDs as unique column names
genus_data = pd.DataFrame(
    data=np.vstack(list(merged_ncbi_dict.values())).T,
    columns=list(merged_ncbi_dict.keys()),
    index=genus_data.index
)

# Step 3: Use NCBI to get the phylogenetic tree and order genera accordingly
tree = ncbi.get_topology(valid_raw_id_list)
tree_ascii = tree.get_ascii(attributes=["taxid"])
output_path_tree = "./IBD/phylogenTree_data/phylogenetic_tree.txt"
with open(output_path_tree, "w") as file:
    file.write(tree_ascii)

# Get the postorder traversal leaf order
postorder = []
for node in tree.traverse(strategy='postorder'):
    if node.is_leaf():
        postorder.append(node.name)

# Reorder the data based on the tree order
ordered_genus_data_p = genus_data[postorder]

# Row-wise normalization function
def normalize_to_one(df):
    numeric_df = df.copy()
    numeric_df = numeric_df.apply(pd.to_numeric, errors='coerce')  # Ensure all values are numeric
    row_sums = numeric_df.sum(axis=1)  # Compute row sums
    normalized_df = numeric_df.div(row_sums, axis=0)  # Normalize each row
    normalized_df.fillna(0, inplace=True)  # Fill possible NaN values with 0
    return normalized_df

# Normalize the reordered data
normalized_genus_data_p = normalize_to_one(ordered_genus_data_p)

# Save the normalized data
output_path_p = './IBD/phylogenTree_data/phylogenTree_p_Train.csv'
normalized_genus_data_p.to_csv(output_path_p)

print(f"Phylogenetic tree saved to {output_path_tree}")
print(f"Normalized data saved to {output_path_p}")
