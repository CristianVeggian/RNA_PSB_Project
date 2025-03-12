import subprocess
import csv
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

def run_experiment(dataset, model, seed):
    #subprocess.run(['pip', 'install', 'requirements.txt'], capture_output=True, text=True)
    cmd = [sys.executable, "run.py", dataset, model, str(seed)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout, result.stderr

if __name__ == "__main__":

    NUM_CORES = 3

    setups = []
    with open("setup.csv", newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            setups.append((row["dataset"], row["model"], int(row["seed"])))

    # Executor para escalonar os experimentos
    with ThreadPoolExecutor(max_workers=NUM_CORES) as executor:
        futures = {executor.submit(run_experiment, *setup): setup for setup in setups}

        for future in as_completed(futures):
            setup = futures[future]
            try:
                output, error = future.result()
                print(f"Setup {setup} completed with output:\n{output}")
                if error:
                    print(f"Setup {setup} had errors:\n{error}")
            except Exception as exc:
                print(f"Setup {setup} generated an exception: {exc}")
