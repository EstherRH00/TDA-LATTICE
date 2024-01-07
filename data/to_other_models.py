import csv
import pandas as pd
import json


# 0. item_list & user_list to csv
def txt_to_csv(input_file, output_file, delimiter='\t', field_1 = 'asin', field_2='value'):
    with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
        csv_writer = csv.DictWriter(outfile, delimiter="\t", fieldnames=[field_1, field_2])

        # Write the CSV header
        csv_writer.writeheader()

        for line in infile:
            # Split the line into fields using the specified delimiter
            fields = line.strip().split(delimiter)

            # Create a dictionary to map field names to values
            row_dict = dict(zip([field_1, field_2], fields))

            # Write the row to the CSV file
            csv_writer.writerow(row_dict)

def to_other_models(name = 'Baby'):
    folder = './' + name + '/'

    txt_to_csv(folder+'5-core/item_list.txt',folder + 'i_id_mapping.csv', field_2='itemID')
    txt_to_csv(folder+'5-core/user_list.txt',folder + 'u_id_mapping.csv', field_2 = 'userID' )

    # 1. Replace ID only of the valid ones!

    df = pd.read_csv(folder + "meta-data/ratings_%s.csv" % name, names=['userID', 'itemID', 'rating', 'timestamp'], header=None)

    mapping_df = pd.read_csv(folder + 'u_id_mapping.csv', sep='\t',  names=['userID', 'newUserID'], header=0)

    merged_df = pd.merge(df, mapping_df, how='outer', on='userID')
    merged_df = merged_df.dropna()
    merged_df = merged_df.drop(columns=['userID'])


    mapping_df = pd.read_csv(folder + 'i_id_mapping.csv', sep='\t',  names=['itemID', 'newItemID'], header=0)

    merged_df = pd.merge(merged_df, mapping_df, how='outer', on='itemID')
    merged_df = merged_df.dropna()
    merged_df = merged_df.drop(columns=['itemID'])

    merged_df = merged_df.rename(columns={"newItemID": "itemID", "newUserID": "userID"})
    merged_df['userID'] = merged_df['userID'].astype(int)
    merged_df['itemID'] = merged_df['itemID'].astype(int)

    merged_df = merged_df[['userID', 'itemID', 'rating', 'timestamp']]

    # 0 si t rain, 1 si val, 2 test
    merged_df['x_label'] = 0

    with open(folder + '5-core/test.json', 'r') as json_file:
        test_vals = json.load(json_file)

    for key, values in test_vals.items():
        for val in values:
            row = merged_df[(merged_df['userID'] == int(key)) & (merged_df['itemID'] == int(val))].index
            merged_df.at[row[0], 'x_label'] = 2

    with open(folder + '5-core/val.json', 'r') as json_file:
        val_vals = json.load(json_file)

    for key, values in val_vals.items():
        for val in values:
            row = merged_df[(merged_df['userID'] == int(key)) & (merged_df['itemID'] == int(val))].index
            merged_df.at[row[0], 'x_label'] = 1

    merged_df.to_csv(folder + name + '.inter', sep='\t', index=False)