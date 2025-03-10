#############################################
# SAVE/LOAD SIMULATION DATA FUNCTIONS
#############################################
import pickle
def save_simulation_data(filename, data):
    with open(filename, "wb") as f:
        pickle.dump(data, f)

def load_simulation_data(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)