from sklearn.model_selection import train_test_split

def stratified_split(df, test_size=0.2, random_state=42):
    strata = income_categories(df)
    train_set, test_set = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=strata
    )
    return train_set, test_set