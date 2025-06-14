# Custom Dataset Integration Guide

This guide provides step-by-step instructions for training the PixelRec model on your custom dataset. Following these steps will ensure that your data is correctly formatted and processed, allowing you to leverage the full capabilities of the multimodal recommender system.

The process involves three main stages:
1.  **Preparing Your Data**: Formatting your interaction data, item metadata, and image files.
2.  **Configuring Your Project**: Creating a custom YAML configuration file to point to your data and define model parameters.
3.  **Executing the Workflow**: Running the provided scripts in sequence to preprocess, train, and evaluate the model.

---

### 1. Data Preparation

Proper data preparation is critical for the model to understand and learn from your dataset. You will need to prepare three key components: **interactions data**, **item metadata**, and **image files**.

#### **Interactions Data (`interactions.csv`)**

This file contains the history of user-item interactions. At a minimum, it must contain a `user_id` and an `item_id` for each interaction.

* **Required Columns**:
    * `user_id`: A unique identifier for the user.
    * `item_id`: A unique identifier for the item. This ID must correspond to an item in `item_info.csv` and its associated image file.
* **Optional Columns**:
    * `timestamp`: The time of the interaction. Can be used for time-based splitting.

**Example: `interactions.csv`**
```csv
user_id,item_id,timestamp
user_101,item_A,1672531200
user_101,item_C,1672617600
user_202,item_A,1672704000
```

#### **Item Metadata (`item_info.csv`)**

This file contains the features for each unique item in your dataset. The model uses this information to learn multimodal representations.

* **Required Column**:
    * `item_id`: A unique identifier for the item. This must match the `item_id`s used in `interactions.csv`.
* **Feature Columns**:
    * **Text**: Columns containing descriptive text (e.g., `title`, `description`, `category`).
    * **Numerical**: Columns containing numerical values (e.g., `price`, `release_year`).
    * **Categorical**: Columns with a finite set of discrete values (e.g., `brand`, `genre`). These will be encoded by the model.

**Example: `item_info.csv`**
```csv
item_id,title,description,price,category,brand
item_A,"Classic Blue T-Shirt","A 100% cotton t-shirt, perfect for everyday wear.",19.99,"Apparel","PixelWear"
item_B,"Modern Desk Lamp","An LED lamp with adjustable brightness.",45.50,"Home Goods","LightUp"
item_C,"Advanced Python Book","A deep dive into Python programming.",79.00,"Books","CodePress"
```

#### **Image Data**

The model requires an image for each item. The image files should be named according to their `item_id`.

* **Naming Convention**: The image filename must match its corresponding `item_id`. For example, the image for `item_A` should be named `item_A.jpg`.
* **Supported Formats**: Common image formats like `.jpg`, `.png`, and `.jpeg` are supported.
* **Directory Structure**: All images should be stored in a single directory.

**Example File Structure:**
```
/path/to/your/project/
└── data/
    ├── custom_images/
    │   ├── item_A.jpg
    │   ├── item_B.jpg
    │   └── item_C.jpg
    └── custom_info/
        ├── interactions.csv
        └── item_info.csv
```

---

### 2. Configuration Setup

The model's behavior is controlled by a YAML configuration file. The easiest way to create one for your custom dataset is to copy and modify an existing example, such as **`configs/simple_config_example.yaml`**.

Create a new file (e.g., **`my_dataset_config.yaml`**) and update the following critical parameters:

* **Paths**: Update these to point to your custom data directories.
    ```yaml
    data_path: "data/custom_info/" # Directory with your CSV files
    images_path: "data/custom_images/" # Directory with your image files
    data_preprocessed_path: "data/preprocessed/custom/"
    split_path: "data/splits/custom/"
    output_path: "output/custom/"
    ```

* **Modalities**: Specify which columns from your `item_info.csv` to use for each modality.
    ```yaml
    modality:
      text_cols: ['title', 'description']
      categorical_cols: ['category', 'brand']
      numerical_cols: ['price']
    ```

* **Dataset Information**: Ensure the filenames match your files.
    ```yaml
    dataset:
      interactions_filename: "interactions.csv"
      item_info_filename: "item_info.csv"
    ```

---

### 3. Execution Workflow

With your data prepared and your configuration file set up, you can now run the end-to-end pipeline. Execute the following scripts from the command line in the specified order.

Replace `configs/my_dataset_config.yaml` with the path to your configuration file.

1.  **Preprocess Data**
    * Cleans and combines your CSV files into a unified format.
    ```bash
    python scripts/preprocess_data.py --config configs/my_dataset_config.yaml
    ```

2.  **Create Data Splits**
    * Splits the interactions into training, validation, and test sets.
    ```bash
    python scripts/create_splits.py --config configs/my_dataset_config.yaml
    ```

3.  **Extract Encoders (Optional, for categorical features)**
    * If you have categorical features, this script creates encoders for them.
    ```bash
    python scripts/extract_encoders.py --config configs/my_dataset_config.yaml
    ```

4.  **Train the Model**
    * Trains the multimodal recommender on your data.
    ```bash
    python scripts/train.py --config configs/my_dataset_config.yaml
    ```

5.  **Evaluate the Model**
    * Calculates performance metrics on the test set.
    ```bash
    python scripts/evaluate.py --config configs/my_dataset_config.yaml
    ```

6.  **Generate Recommendations**
    * Generates top-K recommendations for users in the test set.
    ```bash
    python scripts/generate_recommendations.py --config configs/my_dataset_config.yaml
    ```

After completing these steps, the results, including evaluation metrics and generated recommendations, will be available in the directory specified by the **`output_path`** in your configuration file.