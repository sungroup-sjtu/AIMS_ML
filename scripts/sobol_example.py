import pickle
from sobol_analyze import sobol_analyze
import sys
sys.path.append('..')
from mdlearn import fitting

# load data and model
model = fitting.PerceptronFitter(None, None, [])
model.regressor.is_gpu = False
model.regressor.load('out-ch/model.pt')
with open("normed_trainx.pkl", 'rb') as file:
    normed_trainx = pickle.load(file)

result = sobol_analyze(model, normed_trainx, 1000)
print(result['S1'])