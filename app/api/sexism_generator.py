import random
import pandas as pd

template = pd.read_csv("data/raw_data/sexism_generator.csv")

#Turn columns into variables containing list of strings
string_list = template.columns.tolist()
for variable_name in string_list:
    globals()[variable_name] = template[variable_name].dropna().astype(str)

#Generate Phrases
def generate_sexism(phrases=1):
    """Generates n sexist phrases.
    Outputs a list of strings.
    """
    template_list = template.sample(phrases)
    def sexist_phrase(templatess):
        for i in template.columns.tolist():
            j = template[i].dropna().sample(1).values[0]
            j = j.strip().lower()
            templatess = templatess.replace(i, j)
        t= templatess.replace("{", "").replace("}", "")
        return t
    phrases = [sexist_phrase(templatess) for templatess in template_list]
    return phrases
