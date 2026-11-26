TASKS = {
    "SearchQuery": "search result",
    "QuestionAnswering": "question answering",
    "FactChecking": "fact checking",
    "CodeRetrieval": "code retrieval",
    "Classification": "classification",
    "Clustering": "clustering",
    "SentenceSimilarity": "sentence similarity",
}

_SYMMETRIC = {"Classification", "Clustering", "SentenceSimilarity"}

def format_query(task, text):
    name = TASKS[task]
    return f"task: {name} | query: {text}"


def format_document(text, title=None):
    t = "none" if title is None else title
    return f"title: {t} | text: {text}"


def is_symmetric(task):
    return task in _SYMMETRIC


def format_pair(task, left, right, right_title=None):
    if is_symmetric(task):
        return (format_query(task, left), format_query(task, right))
    return (format_query(task, left), format_document(right, right_title))


def drop_prefix(formatted, u, p):
    result = []
    for s, ui in zip(formatted, u):
        if ui < p:
            result.append(s.split("| query: ", 1)[1])
        else:
            result.append(s)
    return result
