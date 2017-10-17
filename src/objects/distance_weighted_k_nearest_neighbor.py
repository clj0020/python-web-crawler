from .k_nearest_neighbor import KNearestNeighbor

class DistanceWeightedKNearestNeighbor(KNearestNeighbor):

    def __init__(self, machine_learner, k):
        super().__init__(machine_learner, k)


    def run(self):
        self.machine_learner.client.gui.machine_learner_window.display_message("\nDistance Weighted K-Nearest Neighbor initialized...")
        self.load_dataset()
        
        # self.k_nearest_neighbor(self.k)
