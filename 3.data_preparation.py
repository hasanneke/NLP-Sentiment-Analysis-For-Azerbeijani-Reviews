import re
from sklearn.model_selection import train_test_split
from utility.clean_data_func import clean_and_filter_data


def remove_special_characters(text):
    """Removes special characters from a text string, keeping only words and spaces."""
    pattern = r"[^\w\s]"
    return re.sub(pattern, "", text)


def lowercase(text):
    """Converts a text string to lowercase."""
    return text.lower()


# Load and clean data
print("Loading and cleaning data")
df = clean_and_filter_data("azarbeijan-reviews.csv")
df["content"] = df["content"].apply(lambda x: lowercase(remove_special_characters(x)))

# Separate target and data
print("Separating target and data")
content = df["content"].values.tolist()
score = df["score"].values.tolist()

# Split the dataset into training and testing sets
print("Splitting the dataset into training and testing sets")
train_content, test_content, train_score, test_score = train_test_split(
    content, score, test_size=0.20, random_state=42
)
