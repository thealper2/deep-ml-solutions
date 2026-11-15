def matured_labels(outcomes, as_of):
    """{event_id: opened} for outcomes with label_ts <= as_of."""
    result = {}
    for o in outcomes:
        if o["label_ts"] <= as_of:
            result[o["event_id"]] = o["opened"]
    return result