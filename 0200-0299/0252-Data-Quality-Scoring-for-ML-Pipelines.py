def calculate_data_quality_score(data: list, schema: dict) -> dict:
    """
    Calculate data quality metrics for ML pipeline monitoring.
    
    Args:
        data: list of dictionaries representing rows of data
        schema: dictionary defining expected columns and their types
                {'column_name': {'type': 'numeric'|'categorical'|'boolean', 'nullable': True|False}}
    
    Returns:
        dict with keys: 'completeness', 'type_validity', 'uniqueness_ratio', 'overall_score'
        All values as percentages (0-100), rounded to 2 decimal places.
    """
    if not data:
        return {}

    n_records = len(data)
    n_fields = len(schema)
    total_cells = n_records * n_fields

    completeness_count = 0
    type_valid_count = 0

    for record in data:
        for field, spec in schema.items():
            value = record.get(field)

            if value is not None:
                completeness_count += 1

            is_nullable = spec['nullable']
            data_type = spec['type']

            if value is None:
                if is_nullable:
                    type_valid_count += 1
            else:
                if data_type == 'numeric':
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        type_valid_count += 1
                elif data_type == 'categorical':
                    if isinstance(value, str):
                        type_valid_count += 1
                elif data_type == 'boolean':
                    if isinstance(value, bool):
                        type_valid_count += 1

    completeness = (completeness_count / total_cells) * 100 if total_cells > 0 else 0
    type_validity = (type_valid_count / total_cells) * 100 if total_cells > 0 else 0

    unique_records = set()
    for record in data:
        record_tuple = tuple(sorted(record.items()))
        unique_records.add(record_tuple)

    uniqueness_ratio = (len(unique_records) / n_records) * 100 if n_records > 0 else 0
    overall_score = 0.4 * completeness + 0.4 * type_validity + 0.2 * uniqueness_ratio

    return {
        'completeness': round(completeness, 2),
        'type_validity': round(type_validity, 2),
        'uniqueness_ratio': round(uniqueness_ratio, 2),
        'overall_score': round(overall_score, 2),
    }