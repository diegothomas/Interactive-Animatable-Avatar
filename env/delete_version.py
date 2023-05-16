new_data_lines = []

depend_flg = 0
with open("AITS.yml" , "r") as f:
    for i , data_line in enumerate(f):
        if data_line == "  - pip:\n" or data_line == "prefix:\n" :
            depend_flg = 0

        if depend_flg == 1 :
            new_data_line = "=".join(data_line.split("=")[:-1]) + "\n"
            new_data_lines.append(new_data_line)
        else:
            new_data_lines.append(data_line)

        if data_line == "dependencies:\n" :
            depend_flg = 1
        
        
        

with open("AITS2.yml", mode='w') as f:
    f.writelines(new_data_lines)
