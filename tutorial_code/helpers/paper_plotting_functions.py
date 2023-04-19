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
import tikzplotlib

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


def paper_plots_example1_1(*, alpha_phi, all_fbos, mags):
    """Plotting function for example1_1

    Args:
        alpha_phi (float): 'alpha_Phi' in the shim dictionary from example1_1
        all_fbos (list[np.ndarray]): 'all_fbos' in the stats dictionary from example1_1
        mags (list[np.ndarray]): 'mags' in the stats dictionary from example1_1
    """
    # Plot FBOs for first embedding
    plt.clf()

    plt.plot(
        np.array([x[0] for x in all_fbos])
    )
    plt.title('Flux-bias offsets')
    plt.xlabel('Iteration')
    plt.ylabel(r'$\Phi_i$ ($\Phi_0$)')
    plt.ylim(max(abs(np.array(plt.ylim()))) * np.array([-1, 1]))

    if MAKE_TIKZ_PLOTS:
        fn = f'ex11_aPhi{alpha_phi:.6f}_fbos'
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

    # Plot histograms of mags
    plt.clf()

    M = np.array(mags)
    Y = movmean(M, 10)

    plt.hist(Y[10].ravel(), alpha=0.5, bins=np.arange(-.51, .5, 0.02),
             label=f'first 10 iterations',
             density=True)
    plt.hist(Y[-1].ravel(), alpha=0.5, bins=np.arange(-.51, .5, 0.02),
             label=f'last 10 iterations',
             density=True)

    plt.title('Magnetizations')
    plt.xlabel(r'$\langle s_i \rangle$')
    plt.ylabel('Prob. density')
    plt.legend(frameon=False)

    plt.xlim([-.5, .5])
    plt.ylim((0, plt.ylim()[-1] * 1.4))

    if MAKE_TIKZ_PLOTS:
        fn = f'ex11_aPhi{alpha_phi:.6f}_mag_hist'
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

    # Plot mean abs difference of mags
    plt.clf()

    plt.plot(
        # Plots the mean abs difference in magnetization from one call to the next...
        np.std(np.abs(np.diff(M, axis=0)), axis=(1, 2))
    )
    plt.title(r'Mean jump in $\langle s_i\rangle$')
    plt.xlabel('Iteration')
    plt.ylabel(r'Mean absolute difference')
    plt.ylim([0, .7])

    if MAKE_TIKZ_PLOTS:
        fn = f'ex11_aPhi{alpha_phi:.6f}_mag_diff'
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

    # Plot magstd
    plt.clf()

    plt.plot(np.std(M, axis=(1, 2)))
    plt.title(r'$\sigma$ of qubit magnetizations')
    plt.xlabel('Iteration')
    plt.ylabel(r'$\sigma$')
    plt.ylim([0, .5])

    if MAKE_TIKZ_PLOTS:
        fn = f'ex11_aPhi{alpha_phi:.6f}_mag_std'
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


def paper_plots_example1_2(*, all_couplings, all_fbos):
    """Plotting function for example1_2

    Args:
        all_couplings (list[np.ndarray]): 'all_couplings' in the stats dictionary from example1_2
        all_fbos (list[np.ndarray]): 'all_fbos' in the stats dictionary from example1_2
    """
    # Plot FBOs for first embedding
    plt.clf()

    plt.plot(
        np.array([x[0] for x in all_fbos])
    )
    plt.title('Flux-bias offsets')
    plt.xlabel('Iteration')
    plt.ylabel(r'$\Phi_i$ ($\Phi_0$)')
    plt.ylim(max(abs(np.array(plt.ylim()))) * np.array([-1, 1]))

    if MAKE_TIKZ_PLOTS:
        fn = f'ex12_fbos'
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

    # Plot Js for first embedding
    plt.clf()

    plt.plot(
        np.array([x[0] for x in all_couplings])
    )
    plt.title('Couplings')
    plt.xlabel('Iteration')
    plt.ylabel(r'$J_{i,j}$')

    if MAKE_TIKZ_PLOTS:
        fn = f'ex12_Js'
        code = tikzplotlib.get_tikz_code(
            standalone=True,
            axis_width='5cm', axis_height='5cm',
            float_format='.5g',
            extra_axis_parameters=tikz_axis_parameters + [
                r'yticklabel style={/pgf/number format/.cd,fixed,fixed zerofill,precision=3,},'],
        )
        code = code.replace('\\documentclass{standalone}',
                            '\\documentclass{standalone}\n' + extra_code)

        with open(f'{PATH_TO_PAPER_DIR}/tex/{fn}.tex', "w") as f:
            f.write(code)
    else:
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
    """Plotting function for example2_2.

    Args:
        nominal_couplings (np.ndarray): 'nominal_couplings' in the shim dict from example2_2
        all_couplings (list[np.ndarray]): 'all_couplings' in the stats dictionary from example2_2
        all_fbos (list[np.ndarray]): 'all_fbos' in the stats dictionary from example2_2
        mags (np.ndarray): 'mags' in the stats dictionary from example2_2
        frust (np.ndarray): 'frust' in the stats dictionary from example2_2
    """

    # Plot FBOs for first embedding
    plt.clf()

    plt.plot(
        np.array([x[0] for x in all_fbos])
    )
    plt.title('Flux-bias offsets')
    plt.xlabel('Iteration')
    plt.ylabel(r'$\Phi_i$ ($\Phi_0$)')
    plt.ylim(max(abs(np.array(plt.ylim()))) * np.array([-1, 1]))

    if MAKE_TIKZ_PLOTS:
        fn = f'ex22_fbos'
        code = tikzplotlib.get_tikz_code(
            standalone=True,
            axis_width='4cm', axis_height='5cm',
            float_format='.5g',
            extra_axis_parameters=tikz_axis_parameters +
            [r'y tick label style={/pgf/number format/.cd,sci,precision=5}, scaled y ticks=false,'],
        )
        code = code.replace('\\documentclass{standalone}',
                            '\\documentclass{standalone}\n' + extra_code)

        with open(f'{PATH_TO_PAPER_DIR}/tex/{fn}.tex', "w") as f:
            f.write(code)
    else:
        plt.show()

    # Plot Js for first embedding
    plt.clf()

    plt.plot(
        np.array([x[0] / nominal_couplings for x in all_couplings])
    )
    plt.title('Couplings (relative to nominal)')
    plt.xlabel('Iteration')
    plt.ylabel(r'$J_{i,j}/J_{i,j}^{{\ nominal}}$')

    if MAKE_TIKZ_PLOTS:
        fn = f'ex22_Js'
        code = tikzplotlib.get_tikz_code(
            standalone=True,
            axis_width='4cm', axis_height='5cm',
            float_format='.5g',
            extra_axis_parameters=tikz_axis_parameters + [
                r'yticklabel style={/pgf/number format/.cd,fixed,fixed zerofill,precision=3,},'],
        )
        code = code.replace('\\documentclass{standalone}',
                            '\\documentclass{standalone}\n' + extra_code)

        with open(f'{PATH_TO_PAPER_DIR}/tex/{fn}.tex', "w") as f:
            f.write(code)
    else:
        plt.show()

    plt.clf()

    M = np.array(mags)
    Y = movmean(M, 10)

    plt.plot(range(10, len(Y)), np.std(Y[10:], axis=(1, 2)))
    plt.title(r'$\sigma_m$ (10-iter M.M.)')
    plt.xlabel('Iteration')
    plt.ylabel(r'$\sigma_m$')

    if MAKE_TIKZ_PLOTS:
        fn = f'ex22_mag_std'
        code = tikzplotlib.get_tikz_code(
            standalone=True,
            axis_width='4cm', axis_height='5cm',
            float_format='.5g',
            extra_axis_parameters=tikz_axis_parameters + [
                r'yticklabel style={/pgf/number format/.cd,fixed,fixed zerofill,precision=2,},'],
        )
        code = code.replace('\\documentclass{standalone}',
                            '\\documentclass{standalone}\n' + extra_code)

        with open(f'{PATH_TO_PAPER_DIR}/tex/{fn}.tex', "w") as f:
            f.write(code)
    else:
        plt.show()

    plt.clf()

    M = np.array(frust)
    Y = movmean(M, 10)

    plt.plot(range(10, len(Y)), np.std(Y[10:], axis=(1, 2)))
    plt.title(r'$\sigma_f$ (10-iter M.M.)')
    plt.xlabel('Iteration')
    plt.ylabel(r'$\sigma_f$')

    if MAKE_TIKZ_PLOTS:
        fn = f'ex22_frust_std'
        code = tikzplotlib.get_tikz_code(
            standalone=True,
            axis_width='4cm', axis_height='5cm',
            float_format='.5g',
            extra_axis_parameters=tikz_axis_parameters + [
                r'yticklabel style={/pgf/number format/.cd,fixed,fixed zerofill,precision=3,},',
                r'scaled y ticks=false,'
            ],
        )
        code = code.replace('\\documentclass{standalone}',
                            '\\documentclass{standalone}\n' + extra_code)

        with open(f'{PATH_TO_PAPER_DIR}/tex/{fn}.tex', "w") as f:
            f.write(code)
    else:
        plt.show()


def paper_plots_example3_2(*, halve_boundary_couplers,
                           type_, nominal_couplings, coupler_orbits,
                           all_fbos, all_couplings, mags, frust, all_psi):
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
    plt.clf()

    plt.plot(
        np.array([x[0] for x in all_fbos])[:, :12],
        alpha=0.5
    )
    plt.title('Flux-bias offsets')
    plt.xlabel('Iteration')
    plt.ylabel(r'$\Phi_i$ ($\Phi_0$)')
    plt.ylim(max(abs(np.array(plt.ylim()))) * np.array([-1, 1]))

    if MAKE_TIKZ_PLOTS:
        fn = f'ex32_fbos_{type_}{"_halved" * halve_boundary_couplers}'
        code = tikzplotlib.get_tikz_code(
            standalone=True,
            axis_width='4cm', axis_height='5cm',
            float_format='.5g',
            extra_axis_parameters=tikz_axis_parameters +
            [r'y tick label style={/pgf/number format/.cd,sci,precision=5}, scaled y ticks=false,'],
        )
        code = code.replace('\\documentclass{standalone}',
                            '\\documentclass{standalone}\n' + extra_code)

        with open(f'{PATH_TO_PAPER_DIR}/tex/{fn}.tex', "w") as f:
            f.write(code)
    else:
        plt.show()

    # Plot Js for first embedding
    plt.clf()
    Jdata = np.array([x[0] / nominal_couplings for x in all_couplings])
    if type_ == 'embedded_finite':
        indices = np.array(coupler_orbits) == coupler_orbits[0]
    else:
        indices = np.arange(0, Jdata.shape[1], 5)

    plt.plot(Jdata[:, indices],
             alpha=0.5)
    plt.title('Couplings (relative to nominal)')
    plt.xlabel('Iteration')
    plt.ylabel(r'$J_{i,j}/J_{i,j}^{{\ nominal}}$')

    if MAKE_TIKZ_PLOTS:
        fn = f'ex32_Js_{type_}{"_halved" * halve_boundary_couplers}'
        code = tikzplotlib.get_tikz_code(
            standalone=True,
            axis_width='4cm', axis_height='5cm',
            float_format='.5g',
            extra_axis_parameters=tikz_axis_parameters + [
                r'yticklabel style={/pgf/number format/.cd,fixed,fixed zerofill,precision=2,},'],
        )
        code = code.replace('\\documentclass{standalone}',
                            '\\documentclass{standalone}\n' + extra_code)

        with open(f'{PATH_TO_PAPER_DIR}/tex/{fn}.tex', "w") as f:
            f.write(code)
    else:
        plt.show()

    plt.clf()

    M = np.array(mags)
    Y = movmean(M, 10)

    plt.plot(range(10, len(Y)), np.std(Y[10:], axis=(1, 2)))
    plt.title(r'$\sigma_m$ (10-iter M.M.)')
    plt.xlabel('Iteration')
    plt.ylabel(r'$\sigma_m$')

    if MAKE_TIKZ_PLOTS:
        fn = f'ex32_mag_std_{type_}{"_halved" * halve_boundary_couplers}'
        code = tikzplotlib.get_tikz_code(
            standalone=True,
            axis_width='4cm', axis_height='5cm',
            float_format='.5g',
            extra_axis_parameters=tikz_axis_parameters + [
                r'yticklabel style={/pgf/number format/.cd,fixed,fixed zerofill,precision=2,},',
            ],
        )
        code = code.replace('\\documentclass{standalone}',
                            '\\documentclass{standalone}\n' + extra_code)

        with open(f'{PATH_TO_PAPER_DIR}/tex/{fn}.tex', "w") as f:
            f.write(code)
    else:
        plt.show()

    plt.clf()

    M = np.array(frust)
    Y = movmean(M, 10)

    # Get Y for orbits
    orbits = np.unique(coupler_orbits)
    Y_orbit = np.zeros((Y.shape[0], len(orbits)))

    for iorbit, orbit in enumerate(orbits):
        mymat = Y[:, :, coupler_orbits == orbit]
        Y_orbit[:, iorbit] = np.std(mymat, axis=(1, 2))

    plt.plot(range(10, len(Y)),
             np.mean(Y_orbit[10:], axis=1)
             )

    plt.title(r'$\sigma_f$ (10-iter M.M., per orbit)')
    plt.xlabel('Iteration')
    plt.ylabel(r'$\sigma_f$')

    if MAKE_TIKZ_PLOTS:
        fn = f'ex32_frust_std_{type_}{"_halved" * halve_boundary_couplers}'
        code = tikzplotlib.get_tikz_code(
            standalone=True,
            axis_width='4cm', axis_height='5cm',
            float_format='.5g',
            extra_axis_parameters=tikz_axis_parameters + [
                r'yticklabel style={/pgf/number format/.cd,fixed,fixed zerofill,precision=3,},',
                r'scaled y ticks=false,'
            ],
        )
        code = code.replace('\\documentclass{standalone}',
                            '\\documentclass{standalone}\n' + extra_code)

        with open(f'{PATH_TO_PAPER_DIR}/tex/{fn}.tex', "w") as f:
            f.write(code)
    else:
        plt.show()

    # Plot histogram heatmaps
    psi_data = []
    psi_data.append(np.array([x[0] for x in all_psi[0:100]]))
    psi_data.append(np.array([x[0] for x in all_psi[200:300]]))
    psi_data.append(np.array([x[0] for x in all_psi[700:800]]))

    for ipsi in range(len(psi_data)):
        plt.clf()
        ax = plt.gca()
        psi = psi_data[ipsi]
        x = np.real(psi.ravel())
        y = np.imag(psi.ravel())
        extent = (-2, 2, -1.95, 1.95)
        numbins = 42
        hb = ax.hexbin(x, y, gridsize=numbins, cmap='inferno', extent=extent,
                       norm=matplotlib.colors.Normalize(vmin=0, vmax=80))

        ax.set_title(r'$\psi$')
        cb = plt.gcf().colorbar(hb, ax=ax)
        cb.set_label('count')
        plt.plot([-1 / np.sqrt(3), 1 / np.sqrt(3)], [-1, 1], color='w', linestyle='-')
        plt.plot([-1 / np.sqrt(3), 1 / np.sqrt(3)], [1, -1], color='w', linestyle='-')
        plt.plot([-2 / np.sqrt(3), 2 / np.sqrt(3)], [0, 0], color='w', linestyle='-')
        plt.xlabel(r'Re$(\psi)$')
        plt.ylabel(r'Im$(\psi)$')
        ax.axis([-1.2, 1.2, -1.2, 1.2])
        ax.set_aspect('equal', 'box')

        if MAKE_TIKZ_PLOTS:
            fn = f'ex32_heatmap{ipsi}_{type_}{"_halved" * halve_boundary_couplers}'
            code = tikzplotlib.get_tikz_code(
                standalone=True,
                axis_width='5cm', axis_height='5cm',
                float_format='.5g',
                extra_axis_parameters=tikz_axis_parameters + [
                    r'colorbar style={ytick={0,' + f'{hb.colorbar.norm.vmax}' + r'}}',
                ],
            )
            code = code.replace('\\documentclass{standalone}',
                                '\\documentclass{standalone}\n' + extra_code)

            with open(f'{PATH_TO_PAPER_DIR}/tex/{fn}.tex', "w") as f:
                f.write(code)
        else:
            plt.show()

    plt.clf()
    M = np.array([np.mean(np.abs(x)) for x in all_psi])
    plt.plot(M)
    plt.title(r'$\langle |\psi|\rangle$')
    plt.xlabel('Iteration')
    plt.ylabel(r'$\langle |\psi|\rangle$')
    plt.ylim([0.5, 0.8])

    if MAKE_TIKZ_PLOTS:
        fn = f'ex32_m_{type_}{"_halved" * halve_boundary_couplers}'
        code = tikzplotlib.get_tikz_code(
            standalone=True,
            axis_width='4cm', axis_height='5cm',
            float_format='.5g',
            extra_axis_parameters=tikz_axis_parameters,
        )
        code = code.replace('\\documentclass{standalone}',
                            '\\documentclass{standalone}\n' + extra_code)

        with open(f'{PATH_TO_PAPER_DIR}/tex/{fn}.tex', "w") as f:
            f.write(code)
    else:
        plt.show()
