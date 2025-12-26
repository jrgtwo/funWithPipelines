import pathlib

root_dir = pathlib.Path(r"./models")

def getModel_paths():
    def parsing_func(file_path):
        return str(file_path.parent) + "/" + file_path.name

    filename_list = []

    for file in root_dir.iterdir():
        print(file)
        filename = parsing_func(file)
        filename_list.append(filename)

    return filename_list