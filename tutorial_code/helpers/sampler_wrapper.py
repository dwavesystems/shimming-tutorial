class SamplerWrapper:
    def __init__(self, sampler_type='real', solver=None, **kwargs):
        """
        Initialize the wrapper class with the appropriate sampler.
        :param sampler_type: 'real' for DWaveSampler, 'mock' for MockDWaveSampler
        :param kwargs: Additional parameters to pass to MockDWaveSampler if it's used
        """
        if sampler_type == 'real':
            # Use real hardware sampler, optionally with a specific solver
            from dwave.system import DWaveSampler
            if solver:
                self.sampler = DWaveSampler(solver=solver)
            else:
                self.sampler = DWaveSampler()
        elif sampler_type == 'mock':
            # Use mock sampler with parameters like topology_type and topology_shape
            from dwave.system.testing import MockDWaveSampler
            self.sampler = MockDWaveSampler(**kwargs)
        else:
            raise ValueError("sampler_type must be 'real' or 'mock'")

    def get_sampler(self):
        """
        Return the sampler instance.
        """
        return self.sampler
