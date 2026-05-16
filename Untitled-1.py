# %%
import ecanalytics as ecx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import seaborn as sns
import os
import glob
import deareis
import pyimpspec
from pyimpspec import Circuit, Series, Parallel, Resistor, Capacitor, Warburg, WarburgOpen, ConstantPhaseElement, CircuitBuilder, ZARC

from multiprocessing import Pool
import os

from ecanalytics.src.analysis.api_adapter import APIAdapter

def Nonideal_Randles_Circuit():
    R_s = Resistor(R=50).set_label("s") \
        .set_lower_limits(R=0) \
        .set_upper_limits(R=200)
    R_ct = Resistor(R=30).set_label("ct") \
        .set_lower_limits(R=0) \
        .set_upper_limits(R=100)
    
    # Usually, tau is at 2e-4, n roughly 0.75
    tau, n_dl = 2e-4, 0.75
    Y_dl = np.power(tau, n_dl) / R_ct.get_value("R") # Initial guess for Y based on Brug's formula ~> 10^-5

    Q_dl = ConstantPhaseElement(Y=Y_dl, n=n_dl).set_label("dl") \
        .set_lower_limits(Y=1e-10, n=0.5) \
        .set_upper_limits(Y=1e-2, n=1.0)
    
    W_diff = WarburgOpen(Y=1e-4, n=0.5, B=0.01).set_label("diff") \
        .set_lower_limits(Y=1e-10,n=0.3,B=1e-4) \
        .set_upper_limits(Y=1e-2,n=1,B=3) \
        .set_fixed(Y=False,n=False,B=False)
    
    return Circuit(Series([R_s, Parallel([Q_dl, Series([R_ct, W_diff])])]))


if __name__ == "__main__":

    RQ_mg = ZARC(tau=2e-3).set_label("mg") \
        .set_lower_limits(tau=5e-4) \
        .set_upper_limits(tau=6e-3)
    
    randles = Nonideal_Randles_Circuit()

    eq = Circuit(Series([randles._elements, RQ_mg]))

    group = "control"
    exp = ecx.Experiment("group").load(f"/Users/nathan/Documents/Jülich/Messungen/2026-03-06/eis-data/{group}")
    res = exp.analysis.fit_circuit(eq)

    fitted = exp.analysis.simulate(res)

    ecx.plot.nyquist([exp, fitted], "Test", style="Data Kind", Rspan=500, add_frequency_labels=True)
    plt.show()



    # pyimpspec.fit_circuit(randles, )

    # # R_sol: Resistor = (
    # #     Resistor()
    # #     .set_values(R=55)
    # #     .set_lower_limits(R=10)
    # #     .set_upper_limits(R=120)
    # #     .set_fixed(R=False)
    # #     .set_label("sol")
    # # )
    # # # C_dl: Capacitor = (
    # #     Capacitor()
    # #     .set_values(C=6.18e-6)
    # #     .set_label("dl")
    # #     .set_fixed(C=False)
    # #     .set_lower_limits(C=1e-9)
    # #     .set_upper_limits(C=1e-3)
    # # )
    # # C_dl: ConstantPhaseElement = (
    # #     ConstantPhaseElement()
    # #     .set_values(Y=2e-5,n=0.92)  # Adjusted initial value
    # #     .set_label("dl")
    # #     .set_fixed(Y=False, n=False)
    # #     .set_lower_limits(Y=1e-10, n=0.6)  # Adjusted lower limits
    # #     .set_upper_limits(Y=1e-2, n=1.0)  # Adjusted upper limits
    # # )
    # # R_ct: Resistor = (
    # #     Resistor()
    # #     .set_values(R=60)
    # #     .set_label("ct")
    # #     .set_fixed(R=False)
    # #     .set_lower_limits(R=1)
    # #     .set_upper_limits(R=2000)
    # # )
    # # W_diff: WarburgOpen = (
    # #     WarburgOpen()
    # #     .set_values(Y=1e-4,n=0.5,B=0.01)
    # #     .set_label("diff")
    # #     .set_fixed(Y=False,n=False,B=False)
    # #     .set_lower_limits(Y=1e-10,n=0.3,B=1e-4)
    # #     .set_upper_limits(Y=1e-2,n=1,B=3)
    # # )

    # #Construct fitting circuit from elements here
    # inner_series: Series = Series([R_ct, W_diff])
    # # parallel: Parallel = Parallel([C_dl, inner_series])
    # outer_series: Series = Series([R_sol, parallel])
    # circuit: Circuit = Circuit(outer_series)
    #For more complex circuits increase number
    circuit_str=randles.to_string(10)
    # Default fitting setting are CNLSMethod.AUTO and Weight.AUTO 
    # Check documentation for specific methods/ weights
    # Increase max_nfev if maximum is reached
    fit_settings_SLB = deareis.FitSettings(cdc=circuit_str, method=deareis.CNLSMethod(deareis.CNLSMethod.AUTO), weight=deareis.Weight(deareis.Weight.AUTO), max_nfev=100000,timeout=0)


    for group in ["control", "MG2", "MG5"]:
        exp = ecx.Experiment(group).load(f"/Users/nathan/Documents/Jülich/Messungen/2026-03-06/eis-data/{group}")

        sample_names = exp.data["Sample Name"].unique()
        ds_api = [pyimpspec.DataSet(p.frequencies, p.impedances) for p in APIAdapter.As_Impedance_Payload(exp.data)]

        ppp = []

        for ds, name in zip(ds_api, sample_names):
            res = deareis.fit_circuit(ds, fit_settings_SLB)
            params = res.to_parameters_dataframe()

            if name == "xxx":
                sim = pyimpspec.simulate_spectrum(res.circuit, ds.get_frequencies())

            flat_params = {f"{row.Element}_{row.Parameter}": row.Value for _, row in params.iterrows()}
            ppp.append(flat_params)

            print(f"\n\n{name}:")
            print(params.to_string)

        df = pd.DataFrame(ppp)
        print(f"\n\n{group}:")
        print(df.mean().to_string())
        print(df.std().to_string())
            

        



    # asd = [pyimpspec.DataSet(p.frequencies, p.impedances) for p in APIAdapter.As_Impedance_Payload(exp.data)]

    # # Iterate through each unique filename (if Filename not unique adjust here)
    # for i, data_comp in enumerate(asd):

    #     print(f"\n\n\n RUN {i}\n\n")

    #     results = deareis.fit_circuit(data_comp, fit_settings_SLB)

    #     # Extract fitting results
    #     parameters = results.to_parameters_dataframe()
    #     statistics = results.to_statistics_dataframe()

    #     Cdl_1khz_fit= parameters.at[1, 'Value']*((2*np.pi*1000)**(parameters.at[2, 'Value']-1))
    #     R_sol_fit=parameters.at[0, 'Value']
    #     C_dl_Y_fit=parameters.at[1, 'Value']
    #     C_dl_n_fit=parameters.at[2, 'Value']
    #     R_ct_fit=parameters.at[3, 'Value']
    #     Wo_B_fit=parameters.at[4, 'Value']
    #     Wo_Y_fit=parameters.at[5, 'Value']
    #     Wo_n_fit=parameters.at[6, 'Value']

    #     print(parameters.to_string())
    #     print(statistics.to_string())


