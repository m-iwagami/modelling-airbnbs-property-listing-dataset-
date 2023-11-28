import os

classification_dir = 'modelling-airbnbs-property-listing-dataset-/models/classification/'
current_dir = os.path.dirname(os.getcwd())
folder_path = os.path.join(current_dir, classification_dir)
folder_name = 'test'
folder_name_dir = os.path.join(folder_path, folder_name)
print(folder_name_dir)
os.mkdir(folder_name_dir)
#print(folder_name_dir)