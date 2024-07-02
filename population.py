class population:
    sus = -1
    inf = {}
    rec = {}
    pn = ''

    def __init__(self, p0: list[int], isn: str='init'):
        self.sus = p0[0]
        self.inf = {}
        self.rec = {}
        self.inf[isn] = p0[1]
        self.rec[isn] = p0[2]
    
    def getPop(self, sn: str='init'):
        return [self.sus, self.inf[sn], self.rec[sn]]
    
    def getAllPop(self):
        return [self.sus] + list(self.inf.values()) + list(self.rec.values())
    
    def getAllPopNms(self):
        return ['S'] + [f'I ({sn})' for sn in self.inf.keys()] + [f'R ({sn})' for sn in self.rec.keys()]
    
    def addPop(self, p: list[int], sn: str='init'):
        self.sus += p[0]
        self.inf[sn] += p[1]
        self.rec[sn] += p[2]