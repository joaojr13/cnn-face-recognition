"""Maiúscula inicial na apresentação; identificadores internos permanecem estáveis."""
def display_name(value):
    return value[:1].upper() + value[1:] if isinstance(value, str) else value


def display_report(value, key=None):
    if isinstance(value, list):
        return [display_name(item) if key in {'labels', 'predictions', 'truth'} else display_report(item) for item in value]
    if isinstance(value, dict):
        return {(display_name(k) if key in {'scores_by_class', 'class_counts', 'classification_report'} else k):
                display_report(v, k) for k, v in value.items()}
    return display_name(value) if key in {'label', 'person', 'training_label', 'top_candidate'} else value
