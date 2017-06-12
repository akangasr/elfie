from matplotlib.backends.backend_pdf import PdfPages

from elfie.utils import write_json_file

def run_and_report(experiment,
                   path,
                   pdf_file="results.pdf",
                   log_file="experiment.json",
                   figsize=(8.27, 11.69)):
    pdf_file = os.path.join(path, pdf_file)
    log_file = os.path.join(path, log_file)

    with PdfPages(pdf_file) as pdf:
        log = experiment(pdf=pdf, figsize=figsize)
        write_json_file(log_file, log)

