import numpy as np
import skfuzzy as fuzz
import skfuzzy.control as ctrl
import matplotlib.pyplot as plt  # Grafiklerin açık kalmasını sağlar

"""
💡 Bulanık Mantık (Fuzzy Logic) Nedir?
Bulanık mantık, klasik mantıktaki "0 veya 1" mantığının yerine, bir değerin
belirli bir üyelik derecesi ile farklı kümelere ait olabileceğini kabul eder.
Örneğin servis kalitesi 7 olabilir ve bu değer hem 'idare' hem de 'iyi' kümelerine
belirli oranlarda üyedir. Bu sayede gerçek hayattaki belirsizlikler modellenebilir.

Bu uygulamada:
- Girdiler: Servis kalitesi ve yemek kalitesi (0-10)
- Çıktı: Bahşiş yüzdesi (0-25%)
- Amaç: Servis ve yemek kalitesine göre uygun bahşiş miktarını bulanık mantık ile hesaplamak
"""

# 1️⃣ Girdi ve çıktı değişkenlerini tanımlama
# Antecedent -> Girdi (input)
# Consequent -> Çıktı (output)


# 2️⃣ Üyelik fonksiyonlarını tanımlama
"""
Üyelik Fonksiyonu (Membership Function): Bir girdinin hangi bulanık kümeye ne kadar
ait olduğunu gösterir. 0 = hiç ait değil, 1 = tamamen ait.
trimf -> üçgen üyelik fonksiyonu (minimum, tepe, maksimum)
"""

# Servis için üyelikler
servis['kotu'] = fuzz.trimf(servis.universe, [0, 0, 5])    # kötü servis
servis['idare'] = fuzz.trimf(servis.universe, [0, 5, 10])  # idare eder servis
servis['iyi'] = fuzz.trimf(servis.universe, [5, 10, 10])   # iyi servis

# Yemek için üyelikler
yemek['kotu'] = fuzz.trimf(yemek.universe, [0, 0, 5])      # kötü yemek
yemek['idare'] = fuzz.trimf(yemek.universe, [0, 5, 10])    # idare eder yemek
yemek['lezzetli'] = fuzz.trimf(yemek.universe, [5, 10, 10])# lezzetli yemek

# Bahşiş için üyelikler
bahsis['dusuk'] = fuzz.trimf(bahsis.universe, [0, 0, 13])   # düşük bahşiş
bahsis['orta'] = fuzz.trimf(bahsis.universe, [0, 13, 25])   # orta bahşiş
bahsis['yuksek'] = fuzz.trimf(bahsis.universe, [13, 25, 25])# yüksek bahşiş

# 3️⃣ Üyelik fonksiyonlarını görselleştirme
# view() fonksiyonu grafikleri açar
servis.view()
yemek.view()
bahsis.view()
plt.show()  # Bu satır olmadan grafik hemen kapanır

# 4️⃣ Bulanık kuralları tanımlama
"""
Kurallar (Rules): Eğer-ise mantığıyla belirlenir
- kural1: Eğer servis iyi veya yemek lezzetli ise, bahşiş yüksek
- kural2: Eğer servis idare eder ise, bahşiş orta
- kural3: Eğer servis kötü veya yemek kötü ise, bahşiş düşük
"""
kural1 = ctrl.Rule(servis['iyi'] | yemek['lezzetli'], bahsis['yuksek'])
kural2 = ctrl.Rule(servis['idare'], bahsis['orta'])
kural3 = ctrl.Rule(servis['kotu'] | yemek['kotu'], bahsis['dusuk'])

# 5️⃣ Kontrol sistemi oluşturma
"""
ControlSystem: Kuralları ve girdileri alır
ControlSystemSimulation: Girdiler verildiğinde çıktıyı hesaplar
"""
bahsis_ctrl = ctrl.ControlSystem([kural1, kural2, kural3])
bahsis_sim = ctrl.ControlSystemSimulation(bahsis_ctrl)

# 6️⃣ Örnek veri seti
# Her satır [servis puanı, yemek puanı]
veri_seti = [
    [2, 3],   # kötü servis ve kötü yemek
    [5, 6],   # idare eder servis ve idare eder yemek
    [9, 8],   # iyi servis ve lezzetli yemek
    [7, 2],   # iyi servis ama kötü yemek
    [3, 9],   # kötü servis ama lezzetli yemek
]

# 7️⃣ Bahşiş hesaplama
for veri in veri_seti:
    servis_puani, yemek_puani = veri
    
    # Girdileri bulanık sisteme ver
    bahsis_sim.input['servis'] = servis_puani
    bahsis_sim.input['yemek'] = yemek_puani
    
    # Hesaplama yap (defuzzification)
    """
    Defuzzification: Bulanık çıktıyı tek bir sayısal değere çevirir.
    Bu sayede bahşişin yüzdesi hesaplanabilir.
    """
    bahsis_sim.compute()
    
    # Sonucu yazdır
    print(f"Servis: {servis_puani}, Yemek: {yemek_puani} => Bahşiş: %{bahsis_sim.output['bahsis']:.2f}")
