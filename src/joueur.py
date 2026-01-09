class joueur : 
   
    def __init__(self, PV_initiaux = 50):
            self.relance_boost_PM = 0
            self.relance_pousse = 0
            self.PM_INITIAUX = 1
            self.PM = 1
            self.PV = PV_initiaux
            self.PV_initiaux = PV_initiaux


    def reset(self):
        self.relance_boost_PM = 0
        self.relance_pousse = 0
        self.PM_INITIAUX = 1
        self.PM = 1
        self.PV = self.PV_initiaux


    def peut_avancer(self):
        return self.PM>0

    def estMort(self):
        return self.PV < 1

    def avancer(self):
        if(self.peut_avancer):
           self.PM-=1
        else : 
            print("Vous n'avez plus de PM")

    def boost_PM(self):
        if(self.relance_boost_PM==0):
            self.PM +=1
            self.relance_boost_PM=10
        else:
            print(f"le sort boost_PM n'est pas accessible avant {self.relance_boost_PM} tours")

    def passe_tour(self):
        self.PV-=1
        self.PM=self.PM_INITIAUX
        if(self.relance_boost_PM>0):
            self.relance_boost_PM-=1
        if(self.relance_pousse>0):
            self.relance_pousse-=1

        