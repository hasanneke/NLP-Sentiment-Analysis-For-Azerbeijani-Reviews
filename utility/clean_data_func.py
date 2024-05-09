import fasttext
import pandas as pd

# Load the pre-trained model
model_path = (
    "./utility/lid.176.bin"  # Make sure to download this model and update the path
)
model = fasttext.load_model(model_path)


def is_azeri_fasttext(text):
    # Convert text to string to handle any non-string data types
    text = str(text)

    # Clean the text by replacing newline characters
    text = text.replace("\n", " ").replace("\r", " ")

    # Ensure the text is a single line and not empty
    if text.strip() == "":
        return False
    # Predict the language of the text
    predictions = model.predict(text, k=1)  # k=1 returns the top 1 prediction
    lang = predictions[0][0].split("__")[-1]  # language code (e.g., 'az')
    return lang == "az"


def clean_and_filter_data(data_path):
    """Loads the dataset, cleans the content, filters for Azerbaijani text, and returns the DataFrame.

    Args:
        data_path (str): Path to the CSV file.
        model (fasttext.FastText): Pre-trained FastText model.

    Returns:
        pandas.DataFrame: The cleaned and filtered DataFrame.
    """

    # Load your data
    df = pd.read_csv(data_path)

    # Display the first few rows (optional)
    print(df.head())

    # Get dataset information (optional)
    print(df.info())

    # Check for missing values
    print(df.isnull().sum())

    # Handle missing values
    df = df.dropna()  # Drop rows with missing values

    # Optionally, fill missing values with 0 or median:
    df["score"] = df["score"].fillna(0)  # Assuming 0 makes sense
    df["upvotes"] = df["upvotes"].fillna(df["upvotes"].median())  # Using the median

    # Convert 'score' and 'upvotes' to integers
    df["score"] = df["score"].astype(int)
    df["upvotes"] = df["upvotes"].astype(int)

    # Outlier handling (basic percentile-based removal)
    high_quantile = df["score"].quantile(0.95)
    df = df[df["score"] < high_quantile]

    # Text cleaning
    df["content"] = df["content"].str.strip()  # Remove leading/trailing whitespace
    df["content"] = (
        df["content"].str.replace("\n", " ").replace("\r", " ")
    )  # Remove newlines
    df["content"] = df["content"].str.replace(
        "[^\w\s]", "", regex=True
    )  # Remove special chars
    df["content"] = df["content"].str.lower()  # Convert to lowercase

    # Filter for Azerbaijani text
    df["is_azeri"] = df["content"].apply(is_azeri_fasttext)
    df_filtered = df[df["is_azeri"]]

    # Optionally, drop the 'is_azeri' column if not needed
    # df_filtered = df_filtered.drop(columns="is_azeri")

    return df_filtered
