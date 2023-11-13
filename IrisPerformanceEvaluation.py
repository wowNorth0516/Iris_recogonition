import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
def IrisPerformanceEvaluation(original_CRRs, reduced_CRRs, dimensions, CRRs, thresholds, FMRs, FNMRs):
    # Print the table of CRRs using different similarity measures
    measures = ['L1 distance measure', 'L2 distance measure', 'Cosine similarity measure']
    table_data = list(zip(measures, original_CRRs, reduced_CRRs))
    headers = ['Similarity measure', 'Original feature set', 'Reduced feature set']
    table = tabulate(table_data, headers = headers, tablefmt = 'grid')
    title = 'CRRs(%) Using Different Similarity Measures'
    print(title.center(len(table.splitlines()[0]), " "))
    print(table)
    # Plot CRRs against different dimensionality
    plt.plot(dimensions, CRRs, marker = 'o')
    plt.title('CRRs(%) Using Features Of Different Dimensionality')
    plt.xlabel('Dimensionality of the feature vector')
    plt.ylabel('Correct recognition rate')
    plt.show()
    # Print the table of FMRs and FNMRs with different threshold values
    table_data = list(zip(thresholds, FMRs, FNMRs))
    headers = ['Thresholds', 'False Match Rate', 'False Non-match Rate']
    table = tabulate(table_data, headers = headers, tablefmt = 'grid')
    title = 'False Match and False Nonmatch Rates with Different Threshold Values'
    print(title.center(len(table.splitlines()[0]), " "))
    print(table)
    # Plot ROC curve for verification mode
    plt.plot(FMRs, FNMRs)
    plt.title('ROC Curve For Verification Mode')
    plt.xlabel('False Match Rate')
    plt.ylabel('False Non-match Rate')
    plt.show()