"""You can run this file using python -m src.main from the root directory"""

from src.data_loaders.rollcall import get_cleaned_rollcall_data


if __name__ == "__main__":
    df = get_cleaned_rollcall_data()
    print(df.shape)
