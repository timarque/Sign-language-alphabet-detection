import pandas as pd
import json

f = open(r"Sign-language-alphabet-detection\valid\_annotations.coco.json")

data_dict = json.load(f)

df_data = pd.DataFrame({"file_name": pd.Series(dtype="str"),
                        "height": pd.Series(dtype="int"),
                        "width": pd.Series(dtype="int"),
                        "letter": pd.Series(dtype="str")})

del data_dict["info"]
del data_dict["licenses"]
categories = data_dict["categories"]
print(categories[26])
for image_dict in data_dict["images"]:
    data_to_insert = []
    data_to_insert.append(image_dict["file_name"])
    data_to_insert.append(int(image_dict["height"]))
    data_to_insert.append(int(image_dict["width"]))
    for image_annotation_dict in data_dict["annotations"]:
        if image_annotation_dict["image_id"] == image_dict["id"]:
            data_to_insert.append(
                categories[int(image_annotation_dict["category_id"])]["name"])
            break
    df_data.loc[-1] = data_to_insert
    df_data.index += 1
df_data.to_csv("valid_label.csv")

    
    
    

