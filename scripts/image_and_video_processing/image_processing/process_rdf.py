
import torch
import scipy.io


def read_rdf(filename):
    print("File not working, to be done")
    return
    # Read file
    with open(filename, 'rb') as file1:
        contents = file1.read()
    return contents
    count = 0
    while True:
        count += 1
    
        # Get next line from file
        line = file1.readline()
    
        # if line is empty
        # end of file is reached
        if not line:
            break
        print("Line{}: {}".format(count, line.strip()))
        
    Lines = file1.readlines()
    
    parameters = {}
    parameters["predata"] =  250
    parameters["postdata"] =  50
    ii = 0
    # Skip header. Maybe it can be read
    while Lines[ii] != "end_of_header":
        ii = ii + 1
    # Read variable list
    while Lines[ii] != "end_of_var_list":
        split = Lines[ii].split(" ")
        parameters[split[0]] =  float(split[1])
        ii = ii + 1
    # 
    raw_data = torch.tensor(list(map(float, Lines[ii:-1])), dtype=torch.float32)
    total_depth = parameters["NF"] + parameters["predata"] + parameters["postdata"]
    raw_data.view(2, parameters["NX"], parameters["NY"], total_depth)
    complex_raw_data = torch.complex(raw_data[0,  ...], raw_data[1, ...])
    complex_raw_data = complex_raw_data[parameters["predata"]:(parameters["NF"] + parameters["predata"]), ...]
    return complex_raw_data

def read_mat(filename, device="cpu"):
    
    # Check extension of file
    if filename.split('.')[-1] != 'mat':
        print("Wrong filename format, this function only reads .mat files. Return empty")
        return
    
    mat = scipy.io.loadmat(filename)

    # Check if a variable exists
    if not mat["root_name"]:
        print("Non valid file, we need a root_name variable. Return empty")
        return
    
    # root_name is used to reconstruct the data array
    root_name = str(mat["root_name"][0])
    # Read all parameters besides data array
    parameters = {}
    for ll  in mat.keys():
        if ll[0] != '_' and root_name not in ll and ll != 'root_name':
            parameters[ll] = float(mat[ll][0][0])
    # Create tensor and read data array
    data_complex_all = torch.zeros((int(parameters["NF"]), int(parameters["NX"]), int(parameters["NY"])), dtype=torch.complex64).to(device)
    len_cut = int(parameters['len_cut'])
    for ii in range(1, int(parameters['num_cuts']) + 1):
        name = root_name + str(ii)
        if ii < int(parameters['num_cuts']) + 1:
            data_complex_all[(ii - 1) * len_cut:ii*len_cut, ...] = torch.from_numpy(mat[name])
        else:
            data_complex_all[(ii - 1) * len_cut:-1, ...] = torch.from_numpy(mat[name])
    return data_complex_all, parameters


def process_complex_data(complex_raw_data, depth, device="cpu"):

    hamming_vec = torch.hamming_window(depth).view(-1, 1, 1).to(device)
    processed_data = torch.mul(complex_raw_data, hamming_vec)
    processed_data = torch.fft.fftshift(torch.fft.fft(processed_data, dim = 0), dim = 0)
    max_val = torch.max(torch.pow(torch.abs(processed_data), 2))
    return processed_data, max_val



