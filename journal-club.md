# Journal Club Session

## Tipps und Aufbau

1. Verstehe das Paper tiefgehend

   - Lies es mehrmals und fokussiere dich auf Motivation, Methode, Experimente
     und Fazit.
   - Identifiziere den Hauptbeitrag und warum er relevant ist.

2. Fasse die Kernpunkte zusammen

   - Wer sind die Autoren
   - Problemstellung & Motivation: Warum wurde das Paper geschrieben?
   - Methode: Erkläre den Ansatz verständlich mit Diagrammen.
   - Ergebnisse: Zeige die wichtigsten Experimente und Erkenntnisse.
   - Stärken & Schwächen: Was ist gut, wo gibt es Einschränkungen?

3. Nutze visuelle Hilfsmittel

   - Weniger Text, mehr Diagramme zur Veranschaulichung.
   - Falls sinnvoll, zeige Code-Beispiele oder Demos.

4. Mache es interaktiv

5. Üben & Timing beachten
   - Kurz halten (10-15 min), um Zeit für Fragen zu lassen.
   - Bereite dich auf kritische Fragen vor und überlege mögliche Antworten.

## Problemstellung & Motivation

### Problem

- Traditionelle Modelle wie gradient-boosted decision trees dominieren zwar den
  Bereich der Tabellendaten, bringen aber Nachteile wie:
  - Begrenzte Generalisierungsfähigkeit
  - Schwer kombinierbare, handgemachte Feature-Engineering-Schritte (was genau
    für anpassungen)

### Motivation

- Es besteht Bedarf an einem flexiblen, robusten und schnellen Modell, das die
  Vorteile von Deep Learning (wie End-to-End-Lernen) mit der Effizienz bei
  kleinen bis mittleren Datensätzen kombiniert.

## Methode

### Vortrainierung auf synthetischen Daten

- Das Modell wird auf Millionen von künstlich generierten, synthetischen
  Datensätzen trainiert, um ein breites Spektrum möglicher Zusammenhänge zu
  erfassen.
- Durch diese künstlich generierten Daten ist es TabPFN möglich eine gewisse
  ungewissheit ohne extra Kosten zu modellieren.

### Transformer-Architektur angepasst an Tabellendaten

- Anders als klassische Transformer, die Sequenzen verarbeiten, wird hier jede
  Tabellenzelle als eigene „Position“ behandelt.
- **Zeilen- (Datenpunkt) und Spalten-Attention:**
  - Abhängigkeiten innerhalb von Zeilen und über Spalten hinweg werden effektiv
    erfasst.

TODO architektur verstehen (evt infos im anderen paper) schema erstellen.

### In-Context Learning

- Bei Inferenz wird der gesamte Datensatz (Trainings- und Testdaten) in einem
  einzigen Durchgang verarbeitet.
- Das Modell passt sein vortrainiertes Wissen direkt an den neuen Datensatz an.

## Ergebnisse

### Performance

- TabPFN übertrifft in Experimenten klassische Modelle wie CatBoost und XGBoost
  sowohl in Genauigkeit als auch in der Vorhersagegeschwindigkeit (z. B. 2.8
  Sekunden vs. mehrere Stunden für optimierte Baselines).

wie wird das modell in etapen trainiert und verwendet

### Robustheit

- Das Modell zeigt hohe Robustheit gegenüber typischen Herausforderungen in
  realen Datensätzen, wie:
  - Fehlenden Werten
  - Uninformativen Features
  - Ausreissern

## Stärken & Schwächen

### Stärken

- Schnelle und effiziente Vorhersagen durch In-Context Learning.
- Hohe Robustheit durch Training auf einer grossen Bandbreite synthetischer
  Datensätze.
- Transformer-Architektur, die komplexe Zusammenhänge in Tabellendaten erfassen
  kann.

### Schwächen

- Mögliche Skalierbarkeitsprobleme bei sehr grossen Datensätzen (mehr als 10.000
  Zeilen/500 Features).
- In bestimmten Anwendungen könnte die Inferenzgeschwindigkeit im Vergleich zu
  hochoptimierten Modellen (z. B. CatBoost) geringer sein.
- Die Komplexität der Methode und der Vortrainierungsphase kann für manche
  Anwendungen eine Hürde darstellen.
