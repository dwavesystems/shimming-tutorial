# Copyright 2023 D-Wave
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import matplotlib
import numpy as np

from matplotlib import pyplot as plt

from helpers.helper_functions import movmean


extra_code = '\\usepackage{sfmath,siunitx}\\usepackage{mathspec}\\setmainfont{TeX Gyre Heros}\\setmathfont(Digits,Latin,Greek){TeX Gyre Heros}'

tikz_axis_parameters = [
    'axis line style=thick',
    'scale only axis',
    'axis background/.style={fill=white}',
    'every axis plot post/.append style={fill opacity=0.3}',
    'line join=bevel',
]

PATH_TO_PAPER_DIR = "../paper_materials/"
MAKE_TIKZ_PLOTS = False
if MAKE_TIKZ_PLOTS:
    raise NotImplementedError("Need to fix incompatibility of tikzplotlib")
    import tikzplotlib


def paper_plots_example1_1(experiment_data_list):
    """Plotting function for example1_1

    Args:
        alpha_phi (float): 'alpha_Phi' in the shim dictionary from example1_1
        all_fbos (list[np.ndarray]): 'all_fbos' in the stats dictionary from example1_1
        mags (list[np.ndarray]): 'mags' in the stats dictionary from example1_1
    """
    num_experiments = len(experiment_data_list)
    # Create a figure with 1 row and 3 columns for side-by-side plots
    fig, axs = plt.subplots(3, 3, figsize=(6 * num_experiments, 15))  # Adjust figsize for width and height
    fig.canvas.manager.set_window_title('Figure 6: Balancing qubits in a FM chain with flux-bias offsets')

    for col, data in enumerate(experiment_data_list):
        alpha_phi = data['alpha_Phi']
        all_fbos = data['all_fbos']
        mags = data['mags']

        # Plot 1: Flux-bias offsets
        axs[0, col].plot(np.array([x[0] for x in all_fbos]))
        axs[0, col].set_title(rf'$\alpha_\phi$={alpha_phi:.1e}')
        axs[0, col].set_xlabel('Iteration')
        axs[0, col].set_ylabel(rf'Flux-bias offsets, $\phi_f$')
        current_ylim = axs[0, col].get_ylim()
        max_ylim = max(abs(current_ylim[0]), abs(current_ylim[1])) * 1.1
        axs[0, col].set_ylim(-max_ylim, max_ylim)

        # Plot 2: Histograms of Magnetizations
        M = np.array(mags)
        Y = movmean(M, 10)
        if len(Y) > 10:
            first_Y = Y[:10]
            last_Y = Y[-10:]
        else:
            first_Y = Y
            last_Y = Y
        axs[1, col].hist(first_Y.ravel(), alpha=0.5, bins=np.arange(-.51, .5, 0.02),
                        label='First 10 Iterations', density=True)
        axs[1, col].hist(last_Y.ravel(), alpha=0.5, bins=np.arange(-.51, .5, 0.02),
                        label='Last 10 Iterations', density=True)
        axs[1, col].set_xlabel(rf'Magnetizations, $\langle s_i \rangle$')
        axs[1, col].set_ylabel('Prob. density')
        axs[1, col].legend(frameon=False)
        axs[1, col].set_xlim([-0.5, 0.5])
        current_ylim_hist = axs[1, col].get_ylim()
        axs[1, col].set_ylim([0, current_ylim_hist[1] * 1.4])

        # Plot 3: Standard deviation of Magnetizations
        std_mags = np.std(M, axis=(1, 2))  # Adjust axes as needed
        axs[2, col].plot(std_mags)
        axs[2, col].set_xlabel('Iteration')
        axs[2, col].set_ylabel(rf'Std Dev of Qubit Magnetizations, $\sigma$')
        axs[2, col].set_ylim([0, 0.5])

    # Adjust layout to prevent overlapping of titles and labels
    plt.tight_layout()

    # Show the plot or save using TikZ if required
    if MAKE_TIKZ_PLOTS:
        for col, data in enumerate(experiment_data_list):
            alpha_phi = data['alpha_Phi']
            file_names = [
                f'ex11_aPhi{alpha_phi:.6f}_fbos',
                f'ex11_aPhi{alpha_phi:.6f}_mag_hist',
                f'ex11_aPhi{alpha_phi:.6f}_mag_std'
            ]

            for row in range(3):
                ax = axs[row, col]
                code = tikzplotlib.get_tikz_code(
                    figure=fig,
                    axis=ax,
                    axis_width='5cm', axis_height='5cm',
                    float_format='.5g',
                    extra_axis_parameters=tikz_axis_parameters
                )
                code = code.replace('\\documentclass{standalone}',
                                    '\\documentclass{standalone}\n' + extra_code)

                with open(f'{PATH_TO_PAPER_DIR}/tex/{file_names[row]}.tex', "w") as f:
                    f.write(code)
    else:
        # Display the canvas with all experiments' plots arranged vertically in columns
        plt.show()


def paper_plots_example1_2(*, all_couplings, all_fbos, mags, frust):
    """
    Plotting function for example1_2, combining Flux-bias offsets and Couplings on a single canvas.

    Args:
        all_couplings (list[np.ndarray]): 'all_couplings' in the stats dictionary from example1_2
        all_fbos (list[np.ndarray]): 'all_fbos' in the stats dictionary from example1_2
    """
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.canvas.manager.set_window_title('Figure 7: Balancing qubits and couplers in a FM chain with flux-bias offsets and coupler adjustments')

    # Plot 1: Flux-bias offsets
    axs[0, 0].plot(np.array([x[0] for x in all_fbos]))
    axs[0, 0].set_xlabel('Iteration')
    axs[0, 0].set_ylabel(rf'Flux-bias offsets, $\Phi_i$ ($\Phi_0$)')
    current_ylim = axs[0, 0].get_ylim()
    max_ylim = max(abs(current_ylim[0]), abs(current_ylim[1])) * 1.1
    axs[0, 0].set_ylim(-max_ylim, max_ylim)

    # Plot 2: Couplings
    axs[0, 1].plot(np.array([x[0] for x in all_couplings]))
    axs[0, 1].set_xlabel('Iteration')
    axs[0, 1].set_ylabel(r'Couplings, $J_{i,j}$')

   # Plot 3: Standard deviation of magnetizations (10-iter moving mean)
    M = np.array(mags)
    Y = movmean(M, 10)
    axs[1, 0].plot(range(10, len(Y)), np.std(Y[10:], axis=(1, 2)))
    axs[1, 0].set_xlabel('Iteration')
    axs[1, 0].set_ylabel(rf'Std Dev of Qubit Magnetizations (10-Iteration Moving Mean), $\sigma_m$')

    # Plot 4: Standard deviation of frustration probability (10-iter moving mean)
    M = np.array(frust)
    Y = movmean(M, 10)
    axs[1, 1].plot(range(10, len(Y)), np.std(Y[10:], axis=(1, 2)))
    axs[1, 1].set_xlabel('Iteration')
    axs[1, 1].set_ylabel(rf'Std Dev of Frustration Prob. (10-Iteration Moving Mean), $\sigma_f$')

    # Adjust layout to prevent overlapping of titles and labels
    plt.tight_layout()

    # TikZ export or show
    if MAKE_TIKZ_PLOTS:
        file_names = ['ex12_fbos', 'ex12_Js', 'ex12_mag_std', 'ex12_frust_std']
        for i, ax in enumerate(axs.flatten()):  # Flatten the 2x2 grid to iterate over all subplots
            fn = file_names[i]
            code = tikzplotlib.get_tikz_code(
                figure=fig,
                axis=ax,
                standalone=True,
                axis_width='5cm', axis_height='5cm',
                float_format='.5g',
                extra_axis_parameters=tikz_axis_parameters + ([
                    r'yticklabel style={/pgf/number format/.cd,fixed,fixed zerofill,precision=3,},']
                    if fn == 'ex12_Js' else []
                ),
            )
            code = code.replace('\\documentclass{standalone}',
                                '\\documentclass{standalone}\n' + extra_code)

            with open(f'{PATH_TO_PAPER_DIR}/tex/{fn}.tex', "w") as f:
                f.write(code)
    else:
        # Display the canvas with all four plots in a 2x2 grid
        plt.show()



def paper_plots_example2_1():
    """Plotting function for example1_1
    """

    if MAKE_TIKZ_PLOTS:
        fn = f'ex2_1_graph_raw'
        code = tikzplotlib.get_tikz_code(
            standalone=True,
            axis_width='12cm', axis_height='4cm',
            float_format='.5g',
            extra_axis_parameters=[
                'every axis plot post/.append style={opacity=1}',
                'scale only axis',
            ],
        )
        code = code.replace('\\documentclass{standalone}',
                            '\\documentclass{standalone}\n' + extra_code)

        with open(f'{PATH_TO_PAPER_DIR}/tex/{fn}.tex', "w") as f:
            f.write(code)
    else:
        plt.show()


def paper_plots_example2_2(*, nominal_couplings, all_fbos, all_couplings, mags, frust):
    """
    Plotting function for example2_2.

    Args:
        nominal_couplings (np.ndarray): 'nominal_couplings' in the shim dict from example2_2
        all_couplings (list[np.ndarray]): 'all_couplings' in the stats dictionary from example2_2
        all_fbos (list[np.ndarray]): 'all_fbos' in the stats dictionary from example2_2
        mags (np.ndarray): 'mags' in the stats dictionary from example2_2
        frust (np.ndarray): 'frust' in the stats dictionary from example2_2
    """
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.canvas.manager.set_window_title('Figure 10: Shimming a frustrated loop')

    # Plot FBOs for first embedding in the first axis (top-left)
    axs[0, 0].plot(np.array([x[0] for x in all_fbos]))
    axs[0, 0].set_xlabel('Iteration')
    axs[0, 0].set_ylabel(rf'Flux-bias offsets, $\Phi_i$ ($\Phi_0$)')
    axs[0, 0].set_ylim(max(abs(np.array(axs[0, 0].get_ylim()))) * np.array([-1, 1]))

    # Plot Js for first embedding in the second axis (top-right)
    axs[0, 1].plot(np.array([x[0] / nominal_couplings for x in all_couplings]))
    axs[0, 1].set_xlabel('Iteration')
    axs[0, 1].set_ylabel(r'Couplings (relative to nominal), $J_{ij}/|J_{ij}|$')

    # Plot std of magnitudes in third axis (bottom-left)
    M = np.array(mags)
    Y = movmean(M, 10)
    axs[1, 0].plot(range(10, len(Y)), np.std(Y[10:], axis=(1, 2)))
    axs[1, 0].set_xlabel('Iteration')
    axs[1, 0].set_ylabel(rf'Std Dev of Qubit Magnetizations (10-Iteration Moving Mean), $\sigma_m$')

    # Plot std of frustration in fourth axis (bottom-right)
    M = np.array(frust)
    Y = movmean(M, 10)
    axs[1, 1].plot(range(10, len(Y)), np.std(Y[10:], axis=(1, 2)))
    axs[1, 1].set_xlabel('Iteration')
    axs[1, 1].set_ylabel(rf'Std Dev of Frustration Prob. (10-Iteration Moving Mean), $\sigma_f$')

    # Adjust layout to prevent overlapping of titles and labels
    plt.tight_layout()

    # Generate TikZ plots or show the canvas
    if MAKE_TIKZ_PLOTS:
        plot_titles = ['FBOs', 'Couplings', 'Mag std', 'Frust std']
        file_names = ['ex22_fbos', 'ex22_Js', 'ex22_mag_std', 'ex22_frust_std']

        for i, ax in enumerate(axs.flatten()):  # Flatten the 2x2 array of axes for iteration
            fn = file_names[i]
            code = tikzplotlib.get_tikz_code(
                figure=fig,  # Use the figure reference to export specific subplots
                axis_width='4cm', axis_height='5cm',
                float_format='.5g',
                extra_axis_parameters=tikz_axis_parameters + [
                    r'yticklabel style={/pgf/number format/.cd,fixed,fixed zerofill,precision=3,},']
            )
            code = code.replace('\\documentclass{standalone}',
                                '\\documentclass{standalone}\n' + extra_code)

            with open(f'{PATH_TO_PAPER_DIR}/tex/{fn}.tex', "w") as f:
                f.write(code)
    else:
        # Display the canvas with all four plots in a 2x2 grid
        plt.show()

def paper_plots_example3_2(*, halve_boundary_couplers,
                           type_, nominal_couplings, coupler_orbits,
                           all_fbos, all_couplings, mags, frust):
    """Plotting function for example3_2.

    Args:
        halve_boundary_couplers (bool): 'halve_boundary_couplers' in the param dictionary from example3_2
        type_ (str): 'type' in the shim dictionary from example3_2
        nominal_couplings (np.ndarray): 'nominal_couplings' in the shim dictionary from example3_2
        coupler_orbits (list[int]): 'coupler_orbits' in the shim dictionary from example3_2
        all_fbos (list[np.ndarray]): 'all_fbos' in the stats dictionary from example3_2
        all_couplings (list[np.ndarray]): 'all_couplings' in the stats dictionary from example3_2
        mags (np.ndarray): 'mags' in the stats dictionary from example3_2
        frust (np.ndarray): 'frust' in the stats dictionary from example3_2
        all_psi (list[np.ndarray]): 'all_psi' in the stats dictionary from example3_2
    """
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    if type_ == 'embedded_finite':
        fig.canvas.manager.set_window_title('Figure 13: Shimming an embedded cylindrical triangular antiferromagnet')
    elif type_ == 'embedded_infinite':
        fig.canvas.manager.set_window_title('Figure 14: Shimming an isotropic, infinite triangular antiferromagnet')
    else:
        fig.canvas.manager.set_window_title('Figure 15: Shimming an isotropic, infinite triangular antiferromagnet (halved boundary couplers)')

    # Plot 1: Flux-bias offsets
    axs[0, 0].plot(np.array([x[0] for x in all_fbos])[:, :12], alpha=0.5)
    axs[0, 0].set_xlabel('Iteration')
    axs[0, 0].set_ylabel(rf'Flux-bias Offsets, $\Phi_i$ ($\Phi_0$)')
    current_ylim = axs[0, 0].get_ylim()
    max_ylim = max(abs(current_ylim[0]), abs(current_ylim[1])) * 1.1
    axs[0, 0].set_ylim(-max_ylim, max_ylim)

    # Plot 2: Couplings (relative to nominal)
    Jdata = np.array([x[0] / nominal_couplings for x in all_couplings])
    if type_ == 'embedded_finite':
        indices = np.array(coupler_orbits) == coupler_orbits[1]
    else:
        indices = np.arange(0, Jdata.shape[1], 5)
    axs[0, 1].plot(Jdata[:, indices], alpha=0.5)
    axs[0, 1].set_xlabel('Iteration')
    axs[0, 1].set_ylabel(r'Couplings (relative to nominal), $J_{ij}/|J_{ij}|$')

    # Plot 3: Standard deviation of m (10-iter moving mean)
    M = np.array(mags)
    Y = movmean(M, 10)
    axs[1, 0].plot(range(10, len(Y)), np.std(Y[10:], axis=(1, 2)))
    axs[1, 0].set_xlabel('Iteration')
    axs[1, 0].set_ylabel(rf'Std Dev of Qubit Magnetizations (10-Iteration Moving Mean), $\sigma_m$')

    # Plot 4: Standard deviation of f (10-iter moving mean per orbit)
    M = np.array(frust)
    Y = movmean(M, 10)
    orbits = np.unique(coupler_orbits)
    Y_orbit = np.zeros((Y.shape[0], len(orbits)))

    for iorbit, orbit in enumerate(orbits):
        mymat = Y[:, :, coupler_orbits == orbit]
        Y_orbit[:, iorbit] = np.std(mymat, axis=(1, 2))

    axs[1, 1].plot(range(10, len(Y)), np.mean(Y_orbit[10:], axis=1))
    axs[1, 1].set_xlabel('Iteration')
    axs[1, 1].set_ylabel(rf'Std Dev of Frustration Prob. (10-Iteration Moving Mean), $\sigma_f$')

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save or show the plot
    if MAKE_TIKZ_PLOTS:
        fn_list = ['flux_bias_offsets', 'couplings_relative', 'sigma_m', 'sigma_f']
        for i, ax in enumerate(axs):
            fn = f'ex32_{fn_list[i]}_{type_}{"_halved" * halve_boundary_couplers}'
            code = tikzplotlib.get_tikz_code(
                figure=fig,
                axis=ax,
                standalone=True,
                axis_width='5cm', axis_height='5cm',
                float_format='.5g',
                extra_axis_parameters=tikz_axis_parameters,
            )
            code = code.replace('\\documentclass{standalone}',
                                '\\documentclass{standalone}\n' + extra_code)
            with open(f'{PATH_TO_PAPER_DIR}/tex/{fn}.tex', "w") as f:
                f.write(code)
    else:
        plt.show()


def paper_plots_example3_2_heatmaps(experiment_data_list):
    num_experiments = len(experiment_data_list)

    # Create a figure with controlled subplot layout
    fig, axs = plt.subplots(
        num_experiments, 3, figsize=(15, 5 * num_experiments),
        gridspec_kw={'width_ratios': [1, 1, 1]}  # Ensure equal width for all columns
    )
     # Add titles for the second and third columns
    axs[0, 1].set_title('Before Shimming', fontsize=14, fontweight='bold')  
    axs[0, 2].set_title('After Shimming', fontsize=14, fontweight='bold')  
    fig.canvas.manager.set_window_title('Figure 16: Complex Order Parameter Ψ')

    # Loop over each experiment
    for row, data in enumerate(experiment_data_list):


        all_psi = data['all_psi']
        type_ = data['type_']
        halve_boundary_couplers = data['halve_boundary_couplers']

        # Plot 1: Mean Magnitude Line Plot (First Column)
        M = np.array([np.mean(np.abs(x)) for x in all_psi])
        axs[row, 0].plot(M)
        axs[row, 0].set_xlabel('Iteration')
        axs[row, 0].set_ylabel(r'$\langle |\psi|\rangle$')
        axs[row, 0].set_ylim([0.5, 0.8])
        axs[row, 0].set_aspect(1.0 / axs[row, 0].get_data_ratio(), adjustable='box')  # Square aspect ratio

        # Prepare data for the two heatmaps
        psi_data = [
            np.array([x[0] for x in all_psi[0:100]]),   # Before shimming
            np.array([x[0] for x in all_psi[700:800]])  # After shimming
        ]

        # Plot 2 and 3: Heatmaps (Second and Third Columns)
        for col, psi in enumerate(psi_data):
            ax = axs[row, col + 1]  # Heatmaps in 2nd and 3rd columns
            x = np.real(psi.ravel())
            y = np.imag(psi.ravel())
            extent = (-2, 2, -1.95, 1.95)
            numbins = 42

            hb = ax.hexbin(
                x, y, gridsize=numbins, cmap='inferno', extent=extent,
                norm=matplotlib.colors.Normalize(vmin=0, vmax=80)
            )

            cb = fig.colorbar(hb, ax=ax, shrink=0.8)  # Adjust colorbar size
            cb.set_label('count')

            # Add reference lines
            ax.plot([-1 / np.sqrt(3), 1 / np.sqrt(3)], [-1, 1], color='w', linestyle='-')
            ax.plot([-1 / np.sqrt(3), 1 / np.sqrt(3)], [1, -1], color='w', linestyle='-')
            ax.plot([-2 / np.sqrt(3), 2 / np.sqrt(3)], [0, 0], color='w', linestyle='-')

            ax.set_xlabel(r'Re$(\psi)$')
            ax.set_ylabel(r'Im$(\psi)$')
            ax.axis([-1.2, 1.2, -1.2, 1.2])
            ax.set_aspect('equal', 'box')  # Square aspect ratio for heatmaps

    # Adjust layout to prevent overlapping
    plt.tight_layout()

    # Display or save the plots
    if MAKE_TIKZ_PLOTS:
        for row, data in enumerate(experiment_data_list):
            type_ = data['type_']
            halve_boundary_couplers = data['halve_boundary_couplers']

            fn = f'ex32_row{row}_{type_}{"_halved" * halve_boundary_couplers}'
            code = tikzplotlib.get_tikz_code(
                standalone=True,
                axis_width='5cm', axis_height='5cm',
                float_format='.5g',
                extra_axis_parameters=tikz_axis_parameters,
            )
            code = code.replace('\\documentclass{standalone}',
                                '\\documentclass{standalone}\n' + extra_code)

            with open(f'{PATH_TO_PAPER_DIR}/tex/{fn}.tex', "w") as f:
                f.write(code)
    else:
        plt.show()
