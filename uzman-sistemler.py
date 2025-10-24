from random import choice
from experta import *


class Isik(Fact):
    """Trafik isiklariyla ilgili genel bilgiler"""
    pass

class KarsidanKarsiyaGecme(KnowledgeEngine):

    @Rule(Isik(renk="yesil"))
    def yesil(self):
        print("Işık yeşil: Güvenli bir şekilde karşıya geçebilirsin. 👣")

    @Rule(Isik(renk="kirmizi"))
    def kirmizi(self):
        print("Işık kırmızı: Dur! Karşıya geçmek tehlikeli. 🚫")

    @Rule(Isik(renk="sari"))
    def sari(self):
        print("Işık sarı: Dikkat et, birazdan kırmızı olacak. Hazır ol ama geçme. ⚠️")


uzman = KarsidanKarsiyaGecme()
uzman.reset()
uzman.declare(Isik(renk=choice(["yesil", "kirmizi", "sari"])))
uzman.run()