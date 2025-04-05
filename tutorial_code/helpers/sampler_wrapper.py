from dimod import SampleSet
from dwave.samplers import SimulatedAnnealingSampler
from dwave.system.temperatures import fluxbias_to_h
from dwave.system.testing import MockDWaveSampler


class ShimmingMockSampler(MockDWaveSampler):
    """Replace the MockSampler by an MCMC sampler with sensitivity to flux_biases.

    We modify the MockSampler routine so that the sampling distribution is
    sensitive to flux_biases (linear fields are modified in proportion to
    flux_biases). Translation of flux_biases into Ising model linear fields
    uses a conversion factor appropriate to single-qubit freezeout.

    flux_biases_baseline can be added as a list of length
    self.properties['num_qubits'].
    This is zero by default, when non-zero the tutorial routines shim away
    the offset by analogy with the noise shimming in QPU solvers.

    Irrelevant warning messages on unsupported (QPU) parameters are suppressed.

    Replacing the default MockSampler sampler routine with Block Gibbs we
    allow a more realistic susceptibility.
    The default topology is chosen to match defect-free Advantage processor
    architectures.
    """

    def __init__(
        self, topology_type="pegasus", topology_shape=[16], flux_biases_baseline=None
    ):
        substitute_sampler = SimulatedAnnealingSampler()
        substitute_kwargs = {
            "beta_range": [0, 3],
            "beta_schedule_type": "linear",
            "num_sweeps": 100,
            "randomize_order": True,
            "proposal_acceptance_criteria": "Gibbs",
        }
        super().__init__(
            topology_type=topology_type,
            topology_shape=topology_shape,
            substitute_sampler=substitute_sampler,
            substitute_kwargs=substitute_kwargs,
        )
        num_qubits = self.properties["num_qubits"]
        if flux_biases_baseline is None:
            self.flux_biases_baseline = [1e-5] * num_qubits
        else:
            self.flux_biases_baseline = flux_biases_baseline
        self.sampler_type = "mock"
        # Added to suppress warnings (not mocked, but irrelevant to tutorial)
        self.mocked_parameters.add("flux_drift_compensation")
        self.mocked_parameters.add("auto_scale")
        self.mocked_parameters.add("readout_thermalization")
        self.mocked_parameters.add("annealing_time")


    def sample(self, bqm, **kwargs):
        """Sample with flux_biases transformed to Ising model linear biases."""

        # Extract flux biases from kwargs (if provided)
        flux_biases = kwargs.pop("flux_biases", None)
        if self.flux_biases_baseline is not None:
            if flux_biases is None:
                flux_biases = self.flux_biases_baseline
            else:
                flux_biases = [
                    sum(fbs) for fbs in zip(flux_biases, self.flux_biases_baseline)
                ]

        # Adjust the BQM to include flux biases
        if flux_biases is None:
            ss = super().sample(bqm=bqm, **kwargs)
        else:
            _bqm = bqm.change_vartype("SPIN", inplace=False)
            flux_to_h_factor = fluxbias_to_h()

            for v in _bqm.variables:
                bias = _bqm.get_linear(v)
                _bqm.set_linear(v, bias + flux_to_h_factor * flux_biases[v])

            ss = super().sample(bqm=_bqm, **kwargs)

            ss.change_vartype(bqm.vartype)

            ss = SampleSet.from_samples_bqm(ss, bqm)  # energy of bqm, not _bqm

        return ss
