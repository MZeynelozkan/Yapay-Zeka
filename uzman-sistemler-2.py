from random import choice
from experta import *


class Belirti(Fact):
    """Diş ağrısı ve ağız sağlığıyla ilgili belirtiler"""
    pass


class DisAgriUzmani(KnowledgeEngine):

    @Rule(Belirti(belirti="fircalama_kanama"))
    def dis_hastaligi(self):
        print("🦷 Diş fırçalarken diş eti kanaması var → Diş hastalığı olabilir. Diş hekimine başvur.")

    @Rule(Belirti(belirti="uzun_sureli_kanama"))
    def diseti_cekilmesi(self):
        print("🩸 Uzun süreli diş eti kanaması var → Diş eti çekilmesi olabilir. Diş hekimine başvur.")

    @Rule(Belirti(belirti="cekilme_ve_kok"))
    def dolgu_yap(self):
        print("🪥 Diş eti çekilmesi var ve kök görünüyor → Dolgu yaptır.")

    @Rule(Belirti(belirti="renk_degisim"))
    def temizle(self):
        print("✨ Dişte renk değişimi var → Dişleri profesyonelce temizlet.")

    @Rule(Belirti(belirti="morarma"))
    def morarma(self):
        print("🟣 Yeni diş çıkarken morarma var → Diş hekimine başvur.")

    @Rule(Belirti(belirti="curuk_agrisiz"))
    def curuk_dolgu(self):
        print("🦷 Dişte ağrısız çürük var → Dolgu yaptır.")

    @Rule(Belirti(belirti="curuk_ileri"))
    def kanal_tedavisi(self):
        print("⚠️ Diş çürüğü ileri seviyede → Kanal tedavisi ve dolgu yaptır.")


# === Uzman sistemi çalıştırma ===
if __name__ == "__main__":
    belirtiler = [
        "fircalama_kanama",
        "uzun_sureli_kanama",
        "cekilme_ve_kok",
        "renk_degisim",
        "morarma",
        "curuk_agrisiz",
        "curuk_ileri"
    ]

    secilen_belirti = choice(belirtiler)

    print(f"\n🧠 Rastgele belirti seçildi: {secilen_belirti}\n")

    uzman = DisAgriUzmani()
    uzman.reset()
    uzman.declare(Belirti(belirti=secilen_belirti))
    uzman.run()
