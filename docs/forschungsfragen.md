# Forschungsfragen

## T1: Feature-Interaktionen

**Frage:** Inwiefern erkennt TabPFN komplexe Feature-Interaktionen im Vergleich
zu Baummodellen wie CatBoost, und bei welchen spezifischen Typen von
Feature-Interaktionen zeigt es Vorteile?.

**Hypothese:** TabPFN kann aufgrund seiner Transformer-Architektur und
In-Context-Learning-Fähigkeiten bestimmte Arten von Feature-Interaktionen besser
erkennen und modellieren als baumbasierte Modelle wie CatBoost

**Experiment:**

1. Erstellung von synthetischen Datensätzen mit kontrollierten
   Feature-Interaktionen

   - Die Datensätze enthalten die folgenden Arten von Interaktionen
     - Polynomiale Interaktion (quadratisch, kubisch)
     - Logische Interaktionen (XOR, AND, OR)
     - Konditionale Abhängikeiten (Schwellenwerte)
     - Räumliche Distanzen (Euklidische Distanz, Manhattan-Distanz)
   - Die Datensätze umfassen drei Grössenklassen: klein (200 Samples), mittel
     (1000 Samples) und gross (5000 Samples)
   - Für jede Interaktionsart werden Datensätze mit 2, 3 und 4 interagierenden
     Features erstellt
   - Das Verhältnis von relevanten zu irrelevanten Features wird variiert (1:0,
     1:1, 1:2, 1:5)
   - Rauschstufen von 0% (kein Rauschen), 10% und 20% werden hinzugefügt

2. Evaluation

   - Neben TabPFN und CatBoost wird ein lineares Modell als Baseline verwendet
   - Performance-Vergleich (AUC, Accuracy) mit zunehmender Komplexität der
     Interaktionen
   - Bessere Performance deutet darauf hin, dass die Interaktionen besser
     erkannt wurden
   - Analyse der benötigten Datenmenge, um Interaktionsmuster zu erlernen
   - Analyse wie stark irrelevante Features die Performance beeinflussen
   - Analyse der Robustheit gegenüber Rauschen

3. Interpretation

   - Verwendung von SHAP-Werten, um zu analysieren, wie beide Modelle
     Interaktionen bewerten
     - SHAP-Werte (SHapley Additive exPlanations) bieten einen mathematisch
       fundierten Ansatz, um zu verstehen, wie Modelle Interaktionen bewerten
   - Visualisierung von 2D-Entscheidungsgrenzen für relevante Feature-Paare
     - Entscheidungsgrenzen zeigen, wie ein Modell den Datenraum in verschiedene
       Klassen einteilt
   - Erstellung von Partial Dependence Plots für interagierende Features
     - PDPs zeigen, wie die durchschnittliche Vorhersage eines Modells sich
       ändert, wenn ein oder mehrere Features variiert werden, während alle
       anderen konstant bleiben.

**notes:**

- Ausgangsdatensatz
- Ein Beispiel für Regression und eines für Klassifikation
- Feature Interaktionen nicht abgebildet so im moment
- Freiheitsgrad
- herausfinden ob TabPFN overfitted
- regularisierungsgrad
- Zusammenhang Freiheitsgrad / Feature interaktion / regularisierung /
  overfitting

## T2: Domänenanpassung durch Fine-Tuning

**Frage:** Inwiefern ermöglichen verschiedene Adapter-basierte
Fine-Tuning-Strategien eine effiziente Anpassung von TabPFN an
domänenspezifische Datensätze, und welche Adapter-Architektur zeigt die beste
Performance-Steigerung im Vergleich zum unveränderten Basismodell?

**Hypothese:** Adapter-basiertes Fine-Tuning, das nur kleine,
aufgabenspezifische Module hinzufügt und anpasst, während die vortrainierten
Gewichte eingefroren bleiben, ermöglicht eine effiziente Anpassung von TabPFN an
spezifische Domänen bei minimaler Trainingsdatenmenge und Rechenaufwand.

**Experiment:**

1. Auswahl und Vorbereitung domänenspezifischer Datensätze

   - Identifikation von 3-5 grossen domänenspezifischen Datensätzen (>100.000
     Samples) aus unterschiedlichen Bereichen:

     - Malicious URLs Datensatz
     - Medizinische Daten
     - Finanzdaten
     - Industrielle Sensordaten
     - Konsumentenverhalten

   - Jeder Datensatz wird aufgeteilt in:

     - 10.000 Samples als Testset für Standard-TabPFN
     - 80.000+ Samples für Fine-Tuning und Validierung
     - 10.000 Samples als finales Evaluierungsset

2. Implementierung verschiedener Adapter-Architekturen

   - Bottleneck-Adapter: Kleine Feed-Forward-Netzwerke parallel zu den
     Hauptschichten
     - Variationen mit unterschiedlichen Reduktionsfaktoren (8, 16, 32)
   - Prefix-Tuning: Trainierbare Präfix-Token am Eingang jeder
     Transformer-Schicht
     - Variationen mit unterschiedlicher Präfix-Länge (5%, 10%, 20% der
       Sequenzlänge)
   - LoRA (Low-Rank Adaptation): Niedrigrangige Anpassungen der
     Aufmerksamkeitsmatrizen
     - Variationen mit unterschiedlichen Rangwerten (r=4, r=8, r=16)

3. Fine-Tuning-Protokoll

   - Progressive Skalierung der Trainingsdatenmenge:
     - 1.000, 5.000, 10.000, 50.000 Samples
   - Es werden nur Adapter-Parameter trainiert, während das Basismodell
     vollständig eingefroren wird

4. Evaluation

   - Performance-Vergleich:
     - Standard-TabPFN (Baseline ohne Fine-Tuning)
     - Verschiedene Adapter-Varianten
     - State-of-the-art domänenspezifische Modelle (z.B. XGBoost, LightGBM)
   - Evaluierungsmetriken:
     - Klassifikation: AUC, F1-Score, Accuracy
     - Regression: RMSE, R2-Score

5. Analyse und Interpretation
   - Lernkurven in Abhängigkeit von Trainingsdatenmenge für verschiedene
     Adapter-Typen
   - Performance-Vergleich über verschiedene Domänen hinweg
