from sklearn.model_selection import train_test_split


def split_customers(df, test_size=0.25, seed=0):
    """Stratified random split on churned; returns (train, test) with fresh indexes."""
    train, test = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df["churned"]
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)