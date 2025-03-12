import sys
import os
import config
import pickle as pickle

from classification import run_classification

# carregar scripts de sys.arg
if(len(sys.argv) != 4):
    print("Uso: run.py [dataset] [modelo] [semente]")
    sys.exit(1)

# verificar se o dataset é válido
if(not(sys.argv[1] in config.VALID_DATASETS)):
    print("Dataset não é válido")
    sys.exit(1)

# verificar se o algoritmo é válido
if(not(sys.argv[2])in config.VALID_MODELS):
    print("Algoritmo não é válido")
    sys.exit(1)

# verificar se a semente é válida
if(not(sys.argv[3].isdigit()) and int(sys.argv[3]) < 0):
    print("Semente não é válida")
    sys.exit(1)
    
dataset = sys.argv[1]
model = sys.argv[2]
seed = int(sys.argv[3])

output_file = os.path.join('results', f"{dataset}_{model}_{seed}.pkl")
print(output_file)

# verificar se o arquivo de saída existe
if(os.path.exists(output_file)):
    print("\n - Arquivo já existe, não rodará novamente")
    sys.exit(1)

# caso contrário, podemos rodar os experimentos
results = run_classification(model_name = model, database = dataset, seed = seed)

# criar diretório de saída se não existir
os.makedirs("results", exist_ok=True)

# salvar resultados
print(" - Salvando resultados")

pickle.dump(results, file = open(output_file, "wb"))

print(" - Finalizado")