import subprocess

if __name__ == '__main__':
    # You will also need to run the Jupyter notebook amplitude_boost_plots.ipynb if you haven't already
    file_paths = ["assemble_response_curve_figure.py",
                  "make_sleep_irregularity_figure.py",
                  "make_melatonin_figure.py"]

    for file_path in file_paths:
        subprocess.run(["python", file_path], check=True)
