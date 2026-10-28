import pandas as pd

# Read the original CSV file
file_path = "./IBD/phylogenTree_data/phylogenTree_Train_CLR.csv"  # Replace with your file path
data = pd.read_csv(file_path, index_col=0)

# If you need to transpose the data (features as rows, samples as columns), uncomment the following line:
# data = data.T

# Compute the number of features and how many features should be in each of the 5 groups
num_features = len(data)
remainder = num_features % 5
padding_needed = 5 - remainder if remainder != 0 else 0

# If the number of features is not divisible by 5, apply an alternating head/tail padding strategy
if padding_needed:
    # Save copies of the first and last features
    first_feature = data.iloc[0]
    last_feature = data.iloc[-1]
    pad_rows = []
    for i in range(padding_needed):
        # Even positions copy the first feature; odd positions copy the last feature
        if i % 2 == 0:
            pad_rows.append(first_feature)
        else:
            pad_rows.append(last_feature)

    # Convert the padded rows into a DataFrame and create new row indices
    pad_df = pd.DataFrame(pad_rows)
    pad_df.index = [f"pad_{i}" for i in range(padding_needed)]

    # Concatenate the padding data with the original data
    data = pd.concat([data, pad_df])

# Recompute the total number of features and the size of each group
num_features = len(data)
group_size = num_features // 5

# Initialize the new feature order list
new_order = []
# Traverse positions within each group, then interleave indices across groups
for i in range(group_size):
    for group in range(5):
        new_order.append(group * group_size + i)

# Reorder the data according to the new order
reordered_data = data.iloc[new_order]

# Export the reordered CSV file
output_path = "./IBD/DL_data/microbiome_Train_group_CLR.csv"  # Replace with your output path
reordered_data.to_csv(output_path)

print(f"Reordered data has been saved to {output_path}")
