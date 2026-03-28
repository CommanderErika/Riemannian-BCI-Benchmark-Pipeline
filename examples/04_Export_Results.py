from pipe_bci_toolkit import MLFlowReporter

if __name__ == "__main__":
    
    # Variables
    TRACKING_URI = "http://127.0.0.1:8080"

    # Export Results
    reporter = MLFlowReporter(TRACKING_URI)

    # Get report in DataFrame
    data = reporter.get_benchmark_dataframe()
    # Or just save locally
    reporter.export_to_csv("./results/result.csv")

    # Print results
    print(data)