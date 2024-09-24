from dimod import BinaryQuadraticModel, SampleSet
from dwave.system.testing import MockDWaveSampler 

class ShimmingMockSampler(MockDWaveSampler):
    def __init__(
        self,
        nodelist=None,
        edgelist=None,
        properties=None,
        broken_nodes=None,
        broken_edges=None,
        topology_type='pegasus',
        topology_shape=[16],
        parameter_warnings=True,
        exact_solver_cutoff=16,
        **config
    ):
        super().__init__(
            nodelist=nodelist,
            edgelist=edgelist,
            properties=properties,
            broken_nodes=broken_nodes,
            broken_edges=broken_edges,
            topology_type=topology_type,
            topology_shape=topology_shape,
            parameter_warnings=parameter_warnings,
            exact_solver_cutoff=exact_solver_cutoff,
            **config
        )

        self.sampler_type = 'mock'
        #self.topology_type = 'pegasus'
        #self.topology_shape = [16]
    
    def get_sampler(self):
        """
        Return the sampler instance.
        """
        return self

    