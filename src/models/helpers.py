from sklearn.model_selection import train_test_split


def split_data(
    df, target_col, test_size=0.15, val_size=0.15, stratify=True, random_state=42
):
    """
    Shuffle and split a dataframe into train, validation, and test sets.
    Returns: train_df, val_df, test_df
    """
    stratify_col = df[target_col] if stratify else None

    # Step 1: hold out test set
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, stratify=stratify_col, random_state=random_state
    )

    # Step 2: split train vs. val
    stratify_col = train_val_df[target_col] if stratify else None
    val_ratio_relative = val_size / (1.0 - test_size)

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_ratio_relative,
        stratify=stratify_col,
        random_state=random_state,
    )

    return train_df, val_df, test_df
