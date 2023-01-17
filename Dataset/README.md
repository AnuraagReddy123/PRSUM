Follow the steps below to prepare the dataset.

Go in the Dataset folder and run the following commands:

1. Run the shrink.py script to shrink the dataset to the desired size. The overall size of the dataset is 333001 pull requests.
```
python shrink.py <enter_size>
```
If you would like to use the entire dataset, enter 333001 as the size or:
```
cp dataset_all.json dataset.json
```

2. Run the filter.py script to filter the dataset. The output would be stored in dataset_filtered.json
```
python filter.py
```

3. Run the make_dataset.py script to make the dataset. This is used to get the corresponding issue titles and build the graphs.
```
chmod +x ../lib/gumtree/gumtree/bin/gumtree
python make_dataset.py
```

4. Run the build_vocab.py script to build the vocabulary
```
python build_vocab.py
```

5. Run the split.py script to split the dataset into train, validation and test sets.
```
python split.py
```

6. Finally run the makefile to remove the intermediate files.
```
make clean
```

There are other util scripts that can be used.
- count.py: This is used to count the number of pull requests present in the given json file
```
python count.py <json_file>
```
- find_data_point.py: This is used to find a particular data point in the dataset. The point is then stored in dataset_testing.json
```
python find_data_point.py <data_point_id, eg: elastic/elasticsearch_37945>
```
- make_graph.py and process_graph.py are used by make_dataset.py to build the graphs. These are not to be used directly.

The dataset is now ready to be used.
