
class IDgen():
    def __init__(self):
        self.currentID = 0
        self.currentGen = 1

    def get_next_ID(self):
        self.currentID += 1

        return self.currentID

    def increase_Gen(self):
        self.currentGen += 1

    def get_Gen(self):
        return self.currentGen


