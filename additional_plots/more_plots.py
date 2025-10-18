from utils.plotting import plot_thetas_stacked

theta_files = [
    "theta_data/thetas_pi2_m50.csv",  # tau_d1
    "theta_data/thetas_strict_m50.csv",  # tau_d1
    "theta_data/thetas_pi6_m50.csv",  # tau_d1
    # "theta_data/thetas_pi2_m200.csv",  # tau_d2
    # "theta_data/thetas_strict_m200.csv",  # tau_d2
    # "theta_data/thetas_pi6_m200.csv",  # tau_d2
    # "theta_data/thetas_pi2_m300.csv",  # tau_d3
    # "theta_data/thetas_strict_m300.csv",  # tau_d3
    # "theta_data/thetas_pi6_m300.csv",  # tau_d3
]

plot_thetas_stacked(theta_files, pdf_path="theta_comparison_m50.pdf")
