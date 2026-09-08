def add_ratio_features(df):
    df_copy = df.copy()
    df_copy['rooms_per_house'] = df_copy['total_rooms'] / df_copy['households']
    df_copy['bedrooms_ratio'] = df_copy['total_bedrooms'] / df_copy['total_rooms']
    df_copy['people_per_house'] = df_copy['population'] / df_copy['households']
    return df_copy