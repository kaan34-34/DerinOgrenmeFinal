# Beyin Tümörü Tespiti için YOLOv8 Tabanlı Derin Öğrenme Projesi

**Ders:** Derin Öğrenme  
**Düzey:** Yüksek Lisans  
**Proje Türü:** Nesne tespiti, medikal görüntü analizi  
**Model:** YOLOv8n  
**Veri Seti:** Brain Tumor YOLO Format Dataset  
**Çalışma Ortamı:** Python, Ultralytics YOLOv8, PyTorch, macOS CPU ortamı  

## Özet

Bu projede manyetik rezonans benzeri beyin görüntüleri üzerinde tümör varlığına ilişkin bölgelerin otomatik olarak tespit edilmesi amaçlanmıştır. Problem, klasik görüntü sınıflandırma yaklaşımından farklı olarak nesne tespiti problemi biçiminde ele alınmıştır. Bu nedenle model yalnızca görüntünün hangi sınıfa ait olduğunu değil, aynı zamanda ilgili bölgenin konumunu da tahmin etmektedir.

Çalışmada YOLOv8 mimarisinin hafif sürümü olan YOLOv8n kullanılmıştır. Veri seti YOLO formatında hazırlanmış, eğitim ve doğrulama klasörleri kontrol edilmiş, eksik etiket dosyaları tamamlanmış ve eğitim, doğrulama, değerlendirme ve tahmin adımlarını otomatikleştiren Python dosyaları oluşturulmuştur. Projenin uçtan uca çalıştığını doğrulamak için kısa süreli eğitim testleri yapılmış, modelin ağırlık dosyası, doğrulama metrikleri ve örnek tahmin çıktıları üretilmiştir.

## 1. Projenin Amacı

Projenin temel amacı, beyin görüntülerinde tümörle ilişkili alanların derin öğrenme tabanlı bir nesne tespit modeli ile otomatik olarak belirlenmesidir. Bu amaç doğrultusunda YOLOv8 kullanılarak iki sınıflı bir tespit sistemi geliştirilmiştir:

- `negative`: Negatif sınıf olarak etiketlenen bölgeler
- `positive`: Pozitif, tümörle ilişkili sınıf olarak etiketlenen bölgeler

Bu proje, medikal görüntü analizinde derin öğrenme modellerinin nasıl hazırlanacağı, eğitileceği, değerlendirileceği ve test görüntüleri üzerinde nasıl kullanılacağına dair bütünlüklü bir uygulama sunmaktadır.

## 2. Hedefler

Proje kapsamında aşağıdaki hedefler gerçekleştirilmiştir:

1. YOLO formatındaki veri seti yapısının incelenmesi.
2. Görüntü ve etiket dosyalarının eşleşme kontrolünün yapılması.
3. Eksik etiket dosyalarının YOLO formatına uygun biçimde tamamlanması.
4. Eğitim için kullanılacak `brain-tumor.yaml` yapılandırma dosyasının hazırlanması.
5. YOLOv8n modeli ile eğitim sürecinin başlatılabilir hale getirilmesi.
6. Eğitim, değerlendirme ve tahmin adımları için ayrı Python scriptlerinin oluşturulması.
7. Veri seti doğrulama scripti ile etiket tutarlılığının kontrol edilmesi.
8. Eğitim çıktılarının, metriklerin ve tahmin sonuçlarının raporlanması.

## 3. Veri Seti

Veri seti proje klasöründe `brain-tumor` adıyla bulunmaktadır. Klasör yapısı YOLO nesne tespiti formatına uygundur:

```text
brain-tumor/
  images/
    train/
    val/
  labels/
    train/
    val/
```

Veri seti doğrulama sonucunda elde edilen istatistikler aşağıdaki gibidir:

| Bölüm | Görüntü Sayısı | Etiket Dosyası | Kutu Sayısı | Boş Etiket Dosyası |
|---|---:|---:|---:|---:|
| Eğitim | 893 | 893 | 925 | 15 |
| Doğrulama | 223 | 223 | 241 | 0 |
| Toplam | 1116 | 1116 | 1166 | 15 |

Sınıf dağılımı:

| Bölüm | Negative Kutu | Positive Kutu |
|---|---:|---:|
| Eğitim | 437 | 488 |
| Doğrulama | 154 | 87 |

İlk incelemede eğitim klasöründe 15 görüntünün karşılık gelen etiket dosyasının bulunmadığı görülmüştür. YOLO formatında boş etiket dosyası, görüntüde işaretlenmiş nesne olmadığı anlamına gelir. Bu nedenle eksik etiketler boş `.txt` dosyaları olarak oluşturulmuş ve veri seti bütünlüğü sağlanmıştır.

## 4. Yöntem

### 4.1 Model Mimarisi

Projede Ultralytics YOLOv8 ailesinin hafif modeli olan YOLOv8n kullanılmıştır. YOLO ailesi, nesne tespiti problemini tek aşamalı bir tahmin süreciyle çözer. Model, görüntüyü girdiye alır ve her nesne için sınıf, güven skoru ve sınırlayıcı kutu koordinatları üretir.

YOLOv8n seçilmesinin nedenleri şunlardır:

- Küçük veri setleri ve hızlı prototipleme için uygundur.
- CPU üzerinde test edilebilir.
- Daha düşük hesaplama maliyetiyle uçtan uca eğitim akışı doğrulanabilir.
- Ultralytics araçları sayesinde eğitim, doğrulama ve tahmin süreçleri standartlaştırılmıştır.

### 4.2 Problem Formülasyonu

Problem iki sınıflı nesne tespiti problemi olarak tanımlanmıştır. Modelin öğrenmesi beklenen çıktı şu bileşenlerden oluşmaktadır:

- Nesnenin sınıfı: `negative` veya `positive`
- Sınırlayıcı kutu koordinatları
- Güven skoru

YOLO etiket formatı aşağıdaki gibidir:

```text
class_id x_center y_center width height
```

Koordinatlar görüntü boyutuna göre normalize edilmiştir ve 0 ile 1 aralığında yer alır.

## 5. Proje Dosyaları

Proje kapsamında aşağıdaki yardımcı dosyalar oluşturulmuştur:

| Dosya | Açıklama |
|---|---|
| `brain-tumor.yaml` | YOLOv8 eğitiminde kullanılan veri seti yapılandırması |
| `requirements.txt` | Proje bağımlılıkları |
| `validate_dataset.py` | Görüntü, etiket ve koordinat doğrulaması |
| `train.py` | YOLOv8n eğitim scripti |
| `evaluate.py` | Eğitilmiş modelin doğrulama seti üzerinde değerlendirilmesi |
| `predict.py` | Eğitilmiş model ile yeni görüntüler üzerinde tahmin yapılması |
| `README.md` | Projeyi çalıştırma yönergeleri |

Scriptler proje klasöründeki `.cache` dizinini kullanacak şekilde hazırlanmıştır. Böylece Ultralytics ve Matplotlib gibi araçların kullanıcı ana dizinine yazma gereksinimi ortadan kaldırılmıştır.

## 6. Çalıştırma Adımları

### 6.1 Sanal Ortam ve Bağımlılıklar

Proje için Python sanal ortamı oluşturulmuş ve gerekli paketler kurulmuştur:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Kurulan temel paketler:

- `ultralytics`
- `torch`
- `torchvision`
- `opencv-python`
- `matplotlib`
- `pyyaml`
- `numpy`

### 6.2 Veri Seti Doğrulama

Veri setini kontrol etmek için:

```bash
.venv/bin/python validate_dataset.py
```

Bu adımda görüntü ve etiket eşleşmeleri, sınıf değerleri ve normalize edilmiş kutu koordinatları kontrol edilir.

### 6.3 Model Eğitimi

Ödevde verilen temel YOLO komutu:

```bash
.venv/bin/yolo detect train data=brain-tumor.yaml model=yolov8n.pt epochs=50 imgsz=640
```

Aynı işlem proje için hazırlanan Python scripti ile şu şekilde çalıştırılabilir:

```bash
.venv/bin/python train.py --model yolov8n.pt --epochs 50 --imgsz 640 --batch 8 --device cpu
```

Eğitim çıktıları varsayılan olarak şu klasöre kaydedilir:

```text
runs/detect/brain_tumor_yolov8n
```

### 6.4 Model Değerlendirme

Eğitim tamamlandıktan sonra en iyi ağırlık dosyası ile değerlendirme yapılır:

```bash
.venv/bin/python evaluate.py --weights runs/detect/brain_tumor_yolov8n/weights/best.pt
```

### 6.5 Tahmin

Doğrulama görüntüleri üzerinde tahmin almak için:

```bash
.venv/bin/python predict.py --weights runs/detect/brain_tumor_yolov8n/weights/best.pt --source brain-tumor/images/val
```

Tahmin çıktıları şu klasöre kaydedilir:

```text
runs/detect/predict
```

## 7. Deneysel Sonuçlar

Proje geliştirme aşamasında iki tür çalıştırma yapılmıştır:

1. **Duman testi:** Modelin, veri setinin ve scriptlerin uçtan uca çalıştığını doğrulamak için `1 epoch`, `64x64` görüntü boyutu ve `batch=1` ile kısa eğitim.
2. **Nihai eğitim:** `50 epoch`, `640x640` görüntü boyutu ve `batch=8` ile yürütülen ana deney.

Nihai eğitim çıktıları `runs/detect/brain_tumor_yolov8n` klasörü altında oluşmuştur. `results.csv` dosyasında 50 epoch kaydı bulunmaktadır. Modelin en iyi ağırlığı `weights/best.pt`, son epoch ağırlığı ise `weights/last.pt` olarak kaydedilmiştir.

### 7.1 Nihai 50 Epoch Deney Sonuçları

| Metrik | Değer |
|---|---:|
| Toplam Epoch | 50 |
| Son Epoch | 50 |
| Son Epoch Train Box Loss | 0.72275 |
| Son Epoch Train Class Loss | 0.78114 |
| Son Epoch Train DFL Loss | 0.92592 |
| Son Epoch Precision | 0.42183 |
| Son Epoch Recall | 0.83326 |
| Son Epoch mAP50 | 0.46390 |
| Son Epoch mAP50-95 | 0.32973 |
| Son Epoch Val Box Loss | 0.94503 |
| Son Epoch Val Class Loss | 1.33419 |
| Son Epoch Val DFL Loss | 1.02164 |

50 epoch sonunda modelin recall değerinin yüksek seviyeye ulaştığı görülmektedir. Bu, modelin doğrulama setindeki nesnelerin önemli bir bölümünü yakalayabildiğini göstermektedir. Precision değerinin recall değerine göre daha düşük kalması ise modelin bazı yanlış pozitif tahminler ürettiğine işaret etmektedir.

### 7.2 En İyi Epoch ve `best.pt` Doğrulaması

`results.csv` üzerinde yapılan incelemede en yüksek mAP50-95 değeri 37. epoch sonunda elde edilmiştir:

| Metrik | Değer |
|---|---:|
| En İyi Epoch | 37 |
| Precision | 0.44836 |
| Recall | 0.86211 |
| mAP50 | 0.49092 |
| mAP50-95 | 0.36469 |
| Val Box Loss | 0.94280 |
| Val Class Loss | 1.08693 |
| Val DFL Loss | 1.01277 |

`weights/best.pt` dosyası ile yapılan ek değerlendirmede genel doğrulama performansı aşağıdaki gibi ölçülmüştür:

| Sınıf | Görüntü | Instance | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|
| Tüm sınıflar | 223 | 241 | 0.455 | 0.847 | 0.491 | 0.365 |
| Negative | 142 | 154 | 0.578 | 0.844 | 0.601 | 0.448 |
| Positive | 81 | 87 | 0.333 | 0.849 | 0.381 | 0.281 |

Bu sonuçlara göre model pozitif sınıfta yüksek recall değerine ulaşmıştır. Bu durum, tümörle ilişkili pozitif bölgelerin büyük bölümünün yakalandığını göstermesi bakımından önemlidir. Bununla birlikte pozitif sınıftaki precision değerinin 0.333 olması, modelin pozitif tahminlerinde yanlış pozitif oranının iyileştirilmeye açık olduğunu göstermektedir.

### 7.3 Duman Testi Sonucu

`runs/detect/smoke_test_clean` klasöründe yapılan 1 epoch, 64px hızlı testte aşağıdaki metrikler elde edilmiştir:

| Metrik | Değer |
|---|---:|
| Precision | 0.02374 |
| Recall | 0.25832 |
| mAP50 | 0.01408 |
| mAP50-95 | 0.00333 |

Bu testin amacı yüksek başarı elde etmek değil, veri yükleme, eğitim, doğrulama, ağırlık kaydetme ve tahmin adımlarının teknik olarak sorunsuz çalıştığını doğrulamaktır.

### 7.4 Üretilen Çıktılar

Eğitim ve değerlendirme sonucunda aşağıdaki dosyalar üretilmiştir:

- `weights/best.pt`: En iyi model ağırlığı
- `weights/last.pt`: Son epoch ağırlığı
- `results.csv`: Epoch bazlı eğitim ve doğrulama metrikleri
- `results.png`: Eğitim eğrileri
- `confusion_matrix.png`: Karmaşıklık matrisi
- `BoxPR_curve.png`: Precision-Recall eğrisi
- `val_batch*_pred.jpg`: Doğrulama görüntüleri üzerinde tahmin örnekleri

## 8. Tartışma

Proje, YOLOv8 tabanlı bir nesne tespiti hattının medikal görüntü verisi üzerinde uygulanabilir olduğunu göstermektedir. Veri seti YOLO formatına uygun olduğu için model eğitimi doğrudan başlatılabilmiştir. Ancak veri setinde bazı görüntülerin etiket dosyalarının eksik olması, eğitim öncesi veri bütünlüğü kontrolünün önemini ortaya koymuştur.

50 epoch eğitim sonunda model, doğrulama setinde genel mAP50 değerini yaklaşık 0.491 ve mAP50-95 değerini yaklaşık 0.365 seviyesine taşımıştır. En iyi mAP50-95 değerinin 37. epochta oluşması, sonraki epochlarda modelin bazı metriklerde dalgalanma yaşadığını göstermektedir. Bu durum küçük ve sınırlı veri setlerinde beklenebilen bir davranıştır.

Pozitif sınıfın klinik açıdan daha kritik olduğu düşünüldüğünde yalnızca genel mAP değerleri değil, pozitif sınıfa ait recall ve precision değerleri de ayrıca incelenmelidir. Bu çalışmada pozitif sınıf recall değerinin 0.849 olması olumlu bir sonuçtur; model pozitif örneklerin büyük kısmını yakalayabilmiştir. Ancak pozitif sınıf precision değerinin 0.333 olması, yanlış pozitif tahminlerin azaltılması için eşik ayarı, veri artırma veya daha güçlü model varyantlarıyla ek deney yapılması gerektiğini göstermektedir.

## 9. Sonuç

Bu çalışma kapsamında beyin tümörü tespiti için YOLOv8 tabanlı bir derin öğrenme projesi başarıyla hazırlanmıştır. Veri seti doğrulanmış, eksik etiket dosyaları tamamlanmış, eğitim yapılandırması oluşturulmuş ve eğitim, değerlendirme ve tahmin scriptleri geliştirilmiştir. Ayrıca modelin uçtan uca çalıştığı kısa eğitim ve tahmin testleri ile doğrulanmıştır.

Nihai 50 epoch eğitim tamamlanmış ve `best.pt` ağırlığı doğrulama seti üzerinde değerlendirilmiştir. Elde edilen sonuçlar, YOLOv8n modelinin veri setindeki pozitif bölgeleri yüksek recall ile tespit edebildiğini, fakat precision değerini artırmak için ek iyileştirmelere ihtiyaç olduğunu göstermektedir. Bu yönüyle çalışma, hem akademik raporlama hem de uygulamalı derin öğrenme proje teslimi için gerekli temel bileşenleri içermektedir.

## 10. Gelecek Çalışmalar

Model performansını geliştirmek için aşağıdaki adımlar önerilmektedir:

1. `yolov8s.pt` veya `yolov8m.pt` gibi daha büyük modellerle karşılaştırmalı deney yapılması.
2. Veri artırma stratejilerinin precision ve recall üzerindeki etkisinin incelenmesi.
3. Pozitif sınıf için güven eşiği analizi yapılarak yanlış pozitif ve yanlış negatif dengesinin optimize edilmesi.
4. Eğitim ve doğrulama görüntülerindeki sınıf dağılımının daha ayrıntılı incelenmesi.
5. K-fold cross validation benzeri daha sağlam değerlendirme yaklaşımlarının uygulanması.
6. Pozitif sınıf için ek veri veya daha dengeli örnekleme stratejilerinin denenmesi.

## 11. Kaynaklar ve Kullanılan Teknolojiler

- Ultralytics YOLOv8
- PyTorch
- OpenCV
- Matplotlib
- NumPy
- YOLO nesne tespiti etiket formatı

## Ek A: Önemli Komutlar

Veri seti doğrulama:

```bash
.venv/bin/python validate_dataset.py
```

50 epoch eğitim:

```bash
.venv/bin/python train.py --model yolov8n.pt --epochs 50 --imgsz 640 --batch 8 --device cpu
```

Değerlendirme:

```bash
.venv/bin/python evaluate.py --weights runs/detect/brain_tumor_yolov8n/weights/best.pt
```

Tahmin:

```bash
.venv/bin/python predict.py --weights runs/detect/brain_tumor_yolov8n/weights/best.pt --source brain-tumor/images/val
```
