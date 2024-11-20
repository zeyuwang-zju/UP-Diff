import os 
import torch as th 



class GroundingCDInput:
    def __init__(self):
        pass 

    def prepare(self, batch):
        """
        batch should be the output from dataset.
        Please define here how to process the batch and prepare the 
        extra input for diffusion model. 
        """

        # return batch["img_A"], batch["label"]
        return th.cat((batch["img_A"], batch["label"]), dim=1)

