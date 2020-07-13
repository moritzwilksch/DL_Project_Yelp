# DL_Project_Yelp

## Setup
1. Download the original YELP dataset [HERE](https://www.yelp.com/dataset)
    - We only need the `yelp_academic_dataset_review.json`
    - copy it into the `/data`
2. Make sure that docker and docker-compose are installed
3. Start the docker container with `docker-compose run app` which will build the image and bring you into the python shell
4. Use function `prepare_raw_data` from `data_loading.py` and pass it the path to the json from (1)
    - It returns a dataframe containing only the review, stars and date such that the dataset fits into memory
    - Save this dataframe as a pickle file (to preserve datatypes and make file handling more easy)
5. Use function `undersample_data` from `data_loading.py` to undersample the dataset (for resource constraints)
    - Pass it the path to the pickle from (2) and specify total number of samples and train-/validation-/test ratio for splitting
    - Trainingset will contain balanced classes, Validation and test set will follow original class distributions
    - Returns a Dict containing six data frames: x_train, y_train, x_val, y_val, x_test, y_test.

