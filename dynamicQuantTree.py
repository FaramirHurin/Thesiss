from extendedQuantTree import Extended_Quant_Tree
from neuralNetworks import NN_man


class DynamicQuantTree:

    def __init__(self, initial_pi_values, lamb, statistic, maxN, minN, alpha, stop, does_restart):
        self.tree = Extended_Quant_Tree(initial_pi_values)
        self.value = 0
        self.lamb = lamb
        self.statistic = statistic
        self.NN = NN_man(len(initial_pi_values), maxN, minN, 300)
        self.NN.train(alpha)
        self.status = 0
        self.trheshold = self.compute_EWMA_threshold()
        self.stop = stop
        self.does_restart = does_restart
        self.buffer = None

    #TODO Computes the threshols for the EWMA used to detect a change
    def compute_EWMA_threshold(self):
        return

    #Compute current value of EMWA based on the value at previous round.
    # Important: it updates the attributes of the object
    def compute_EMWA(self, batch):
        stat = self.statistic(self.tree, batch)
        threshold = self.NN.predict_value(self.tree.pi_values, self.tree.ndata)
        positive = stat > threshold
        self.value = (1 -self.lamb) * self.value + positive * self.lamb
        return self.value

    #Takes care of buffer management and tree update
    def update_tree(self, batch):
        if not self.buffer is None:
            self.tree.modify_histogram(self.buffer)
        self.buffer = batch

    def find_change(self, EMWA_value ,batch):
        #If there is a change we return True
        if EMWA_value > self.threshold:
            if self.does_restart:
                self.restart()
            return True
        else:
            return False

    def restart(self):
        self.tree = None
        self.buffer = None
        self.status = 0
        return

    def playRound(self, batch):
        if self.status == 0:
            self.tree.build_histogram(batch)
            self.status+=1
            return 0, False
        else:
            EMWA = self.compute_EMWA(batch)
            #If we have to stop when we find a change and we think there is one, we exit
            if self.stop and self.find_change(EMWA, batch):
                return EMWA, True
            self.update_tree(batch)
            self.status += 1
            return EMWA, False