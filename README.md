# DL_Project_Yelp

A project by Moritz Wilksch, Sofya Marchenko and Ferdinand Hoske for the Deep Learning course at HPI.

## Setup

1. Download the original YELP dataset [HERE](https://drive.google.com/file/d/1JyhkvYRnpy5mlvI2xUrTEJMdGoEi19PQ/view?usp=sharing)
   - copy the `yelp_academic_dataset_review.json` into the `/data` folder
   - The JSON is only necessary when you want to run Step 4. In case you download the pickle `reviews_optimized.pickle` you can skip this step
2. Make sure that docker and docker-compose are installed
3. Start the docker container with `docker-compose run app` which will build the image and bring you into the python shell. Then all files can be run with `exec(open('filename.py').read())`. The folder `/data` is mounted as Volume. You can shut down the container with `exit()`
   **Recommendation: skip 4 to 8 and proceed right to 9. These steps take ca. 30min running time and 20GB of RAM and another 15GB of disk space**
4. Run `prepare_raw_data.py`
   - It returns a dataframe containing only the review, stars and date such that the dataset fits into memory
   - Save this dataframe as a pickle file (`reviews_optimized.pickle`) (to preserve datatypes and make file handling more easy)
   - Alternatively download `reviews_optimized.pickle` [HERE](https://drive.google.com/file/d/1941uzvFlhurY7j5JtiIoGsuW57onkMT3/view?usp=sharing). Make sure to save it in the `/data` folder.
5. Run `data_loading.py` to undersample the dataset (for resource constraints)
   - Trainingset will contain balanced classes, Validation and test set will follow original class distributions
   - Returns a Dict containing six data frames: x_train, y_train, x_val, y_val, x_test, y_test.
   - Save this dict to disk as `3c_subsampled_data.pickle`.
   - Alternatively download `3c_subsampled_data.pickle` [HERE](https://drive.google.com/file/d/1QuJlZdjBZaicfb9ibXxg1xHueZNzX4gT/view?usp=sharing). Make sure to save it in the `/data` folder.
6. Download the word vector file from fasttext [HERE](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz)
   - copy the .bin file into `/data`
   - The file is only necessary when you want to run Step 7
7. Run `create_embeddings.py`to create the embedding matrix based on the training set.
   - Matrix will only contain the vectors for the words appearing in subsample
   - Saves `embedding_matrix.pickle` and `tokenizer.pickle` to disk. Both are needed for training & testing
   - Alternatively download `embedding_matrix.pickle` [HERE](https://drive.google.com/file/d/1SkrpF67Wbccut8hpEH-K4mlnxobpAlEj/view?usp=sharing) and `tokenizer.pickle` [HERE](https://drive.google.com/file/d/1lUr6QfSDx_neYnBwrO53L_jWnL_O7sd_/view?usp=sharing). Make sure to save them in the `/data` folder.
8. Run `train.py` to train the model yourself.
   - Make sure `embedding_matrix.pickle` and `tokenizer.pickle` (from step 7) are available.
   - Saves `model.hdf5` to disk.
   - Alternatively download `model.hdf5` [HERE](https://drive.google.com/file/d/1McAYruCXwTSRBzxWZJ1k48WvGsN4f0IK/view?usp=sharing). Make sure to save it in the `/data` folder
9. Run `test.py` to examine the model performance on the test set
10. Run `baseline_NaiveBayes.py` to get the performance of Naive Bayes so you can compare the model.
