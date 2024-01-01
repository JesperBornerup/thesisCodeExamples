#Last iteration of script for tagging:

import json
import re
import pickle


# Load data from JSON file
with open('modifiedData.json', 'r') as file:
    myData = json.load(file)
with open('typeEntityPairs.pkl', 'rb') as file:
    entDict = pickle.load(file)



subSubCategories = []
subSubCategories.append("Hjælpemidler/Handling")
subSubCategories.append("Handling")
subSubCategories.append("Social/Familie")
subSubCategories.append("Sted/Placering")
subSubCategories.append("Hjælpermidler/Sted/Placering")
for sub in entDict.keys():
    subSubCategories.append(sub)


if "nerAction" not in myData:
    myData["nerAction"] = {}
if "nerActionSeen" not in myData:
    myData["nerActionSeen"] = []



def makeAnnotation(entity, start, end, newType):
    NERTAG = {
        "start": start,
        "end": end,
        "type": newType,
        "entity": entity
    }
    newAnno = {
        "type": "ner",
        "category": "placeholder",
        "subCategory": "placeholder",
        "subSubCategory": "placeholder",
        "finalSelect": "placeholder",
        "missingNer": NERTAG
    }
    return newAnno

for note, annotations in myData["nerFinal"].items():
    if note in myData["nerActionSeen"]:
        continue
    updated_annotations = []
    words = re.findall(r'\b\w+\b', note)  # Find whole words only
    for idx, word in enumerate(words):
            if word == "Patienten":
                regex_pattern = r'\b' + re.escape(word) + r'\b'
                for match in re.finditer(regex_pattern, note):
                    start = match.start()
                    end = match.end() - 1
                    new_anno = makeAnnotation(word, start, end, "Patient")
                    updated_annotations.append(new_anno)
    for annotation in annotations:
        ner_tag = annotation.get('NERTAG', annotation.get('missingNer'))
        if ner_tag:
            if ner_tag["entity"] in ["liftes", "Liftes"]:
                ner_tag["type"] = "Hjælpemidler/Handling"
        updated_annotations.append(annotation)
    while True:
        print("\n")
        print(note)
        for annotation in updated_annotations:
            ner_tag = annotation.get('NERTAG', annotation.get('missingNer'))
            if ner_tag:
                print(f"Existing NERTAG type: {ner_tag['type']}, entity: {ner_tag['entity']}")

        add_tag = input("Do you want to add a new NER tag? (y/n): ").strip().lower()
        if add_tag != 'y':
            break

        # Logic to add a new NER tag
        words = re.findall(r'\b\w+\b', note)
        for idx, word in enumerate(words):
            print(f'{idx}. {word}')

        user_input = input("Enter the number for the right word: ")
        print("\n")
        indices = map(int, user_input.split())
        selected_words = [words[idx] for idx in indices]
        selected_text = ' '.join(selected_words)
        occurrences = [m.start() for m in re.finditer(re.escape(selected_text), note)]
        if len(occurrences) > 1:
            # If there are multiple occurrences
            print(f"There are {len(occurrences)} occurrences of '{selected_text}'.")
            for i, occ in enumerate(occurrences, start=1):
                print(f"{i}. Occurrence at position {occ}")
            occ_index = int(input("Enter the number for the correct occurrence: ")) - 1
            start = occurrences[occ_index]
        else:
            # If there's only one occurrence
            start = occurrences[0]

        end = start + len(selected_text) - 1

        print("Select a subSubCategory for the new NER tag:")
        print("\n")
        for idx, cat in enumerate(subSubCategories):
            print(f'{idx}. {cat}')
        cat_index = int(input("Enter the number for the right category: "))
        selected_category = subSubCategories[cat_index]
        category = "Placeholder"
        subCategory = "Placeholder"
        subSubCategory = "Placeholder"
        finalSelect = "Placeholder"
        new_ner_tag = {
            "start": start,
            "end": end,
            "type": selected_category,
            "entity": selected_text
        }
        annotation= {
            "category": category,
            "subCategory": subCategory,
            "subSubCategory": subSubCategory,
            "finalSelect": finalSelect,
            "NERTAG": new_ner_tag
        }
        updated_annotations.append({"NERTAG": new_ner_tag})
        myData["missingNerTwo"] += 1
        print(f'New NER tag added: {new_ner_tag}')

    myData["nerAction"][note] = updated_annotations
    myData["nerActionSeen"].append(note)

    # Save progress
    with open('modifiedData.json', 'w') as file:
        json.dump(myData, file, indent=4)

    print(f"Processed {len(myData['nerActionSeen'])} notes.")

print("Data review and modification completed.")
