#This code will show the stratified split of the data

import random
#helper functions
def get_types_in_note(details):
    """
    Returns a list of all types within a given set of details.
    """
    types = []
    for detail in details:
        if 'NERTAG' in detail or 'missingNer' in detail:
            tag_key = detail.get('NERTAG', detail.get('missingNer'))
            annotation_type = tag_key.get('type')
            if annotation_type:
                types.append(annotation_type)
    return types

def get_dominant_type(types):
    """
    Returns the most frequent type within a list of types.
    If there's a tie, returns one randomly.
    """
    if not types:
        return None
    type_counts = {typ: types.count(typ) for typ in set(types)}
    max_count = max(type_counts.values())
    max_types = [t for t, count in type_counts.items() if count == max_count]
    return random.choice(max_types)

def split_data_stratified(allData, num_splits=6):
    """
    Splits the data into num_splits datasets based on types in each note.
    """
    # Initialize counters for each type in each split
    type_counters = {}
    for note, details in allData.items():
        for typ in get_types_in_note(details):
            if typ not in type_counters:
                type_counters[typ] = [0] * (num_splits)

    # Initialize splits
    splits = {i: set() for i in range(num_splits)}
    # Assign notes to splits
    for note, details in allData.items():
        types_in_note = get_types_in_note(details)
        dominant_type = get_dominant_type(types_in_note)
        # Prioritize types that are underrepresented
        underrepresented_types = [typ for typ, counts in type_counters.items() if min(counts) <= 2]
        chosen_type = next((typ for typ in types_in_note if typ in underrepresented_types), dominant_type)
        if chosen_type:
        # Find the split(s) with the least representation of chosen_type
            min_count = min(type_counters[chosen_type])
            candidate_splits = [i for i, count in enumerate(type_counters[chosen_type]) if count == min_count]
            split_index = random.choice(candidate_splits)  # Choose a split index from candidates
            for typ in types_in_note:
                type_counters[typ][split_index] += 1
        else:
            split_index = random.randint(0, num_splits - 1)
            
        splits[split_index].add(note)
    return splits

# splits = split_data_stratified(data)
