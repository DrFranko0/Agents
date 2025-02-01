import json

with open ("soil_comp.json","r") as file:
    data=json.load(file)

with open ("farming_method.json","r") as new:
    new_data=json.load(new)

training_data=[]

for soil_type, details in data["soil_types"].items():
    characteristics = details["characteristics"]
    training_data.append({
        "prompt": f"What are the characteristics of {soil_type} soil?",
        "response": ", ".join([f"{key.replace('_', ' ').capitalize()}: {value}" for key, value in characteristics.items()])
    })
    for crop_type, crops in details["suitable_crops"].items():
        crop_names = [crop["name"] for crop in crops]
        training_data.append({
            "prompt": f"What {crop_type} grow well in {soil_type} soil?",
            "response": ", ".join(crop_names)
        })
        for crop in crops:
            for key, value in crop.items():
                if key != "name":
                    training_data.append({
                        "prompt": f"What is the {key.replace('_', ' ')} of {crop['name']}?",
                        "response": value
                    })

if "farming_methods" in farming_data:
    for method_type, methods in farming_data["farming_methods"].items():
        for method_name, method_details in methods.items():
            training_data.append({
                "prompt": f"What is {method_name.replace('_', ' ')}?",
                "response": method_details["characteristics"]["description"]
            })
            
            characteristics = method_details["characteristics"]
            training_data.append({
                "prompt": f"What are the characteristics of {method_name.replace('_', ' ')}?",
                "response": ", ".join([f"{key.replace('_', ' ').capitalize()}: {value}" for key, value in characteristics.items()])
            })
            
            training_data.append({
                "prompt": f"What are the advantages of {method_name.replace('_', ' ')}?",
                "response": ", ".join(method_details["advantages"])
            })
            
            training_data.append({
                "prompt": f"What are the disadvantages of {method_name.replace('_', ' ')}?",
                "response": ", ".join(method_details["disadvantages"])
            })
            
            training_data.append({
                "prompt": f"What is the typical farm size for {method_name.replace('_', ' ')}?",
                "response": method_details["typical_farm_size"]
            })
            
            training_data.append({
                "prompt": f"In which regions is {method_name.replace('_', ' ')} used?",
                "response": ", ".join(method_details["regions"])
            })


with open("training_data.json", "w") as output_file:
    json.dump(training_data, output_file, indent=4)