def split_features_labels(df):
    X = df.drop(columns=['median_house_value'])
    y = df['median_house_value']
    return X, y