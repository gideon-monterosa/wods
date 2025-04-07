# Forschungsfragen

## T1: Feature-Interaktionen

**Frage:** Wie gut erkennt TabPFN Zusammenhänge zwischen mehreren Features im
Vergleich zu Baummodellen wie CatBoost?

**Hypothese:** TabPFN kann Situationen, in denen Features zusammenarbeiten, um
Ergebnisse vorherzusagen, besser identifizieren als herkömmliche Methoden.

**Experimente:**

- Testdatensätze erstellen, bei denen das Ergebnis von Kombinationen aus 2, 3
  oder 4 Features abhängt
- Vergleichen, wie genau TabPFN und CatBoost diese Ergebnisse vorhersagen
- Visualisierungen verwenden, um zu analysieren, wie jedes Modell
  Feature-Beziehungen versteht

**Notizen:**

- Statistisch anspruchsvoll
- Auf bestimmte Abhängigkeiten beschränken
- Die Untersuchte Art von Zusammenhang muss relevant sein
- Mögliche abhängigkeitsszenarion von unabhängigen Variablen

## T2: Effektivität von Ensembles

**Frage:** Warum funktionieren TabPFNs mehrere Modelle gut zusammen und welche
Kombinationen funktionieren am besten?

**Hypothese:** Verschiedene Vorverarbeitungsmethoden und Datenpermutationen
führen dazu, dass einzelne TabPFN-Modelle unterschiedliche Muster in den Daten
erkennen, die sich beim Kombinieren ergänzen und die Vorhersagen verbessern.

**Experimente:**

- TabPFN default und Post-Hoc-Ensemble (PHE) vergleichen.
- Messen, wie ähnlich oder unterschiedlich die Vorhersagen der einzelnen Modelle
  im Ensemble sind
- Leistung testen, wenn bestimmte Vorverarbeitungsschritte entfernt werden
- Herausfinden, welche Modellkombinationen den grössten Leistungsschub bringen

**Notizen:**

- Konkreter werden
- Mehr auf TabPFN beziehen

## T3: Domänenanpassung durch Fine-Tuning

**Frage:** Welche Fine-Tuning-Strategie eignet sich am besten, um TabPFN an
spezifische Datendomänen anzupassen?

**Hypothese:** Verschiedene Fine-Tuning-Ansätze zeigen unterschiedliche
Effektivität, abhängig von der Ähnlichkeit zwischen der Zieldomäne und der
synthetischen Trainingsverteilung.

**Experimente:**

- Diverse Fine-Tuning-Methoden auf TabPFN anwenden
- Verschiedene Fine-Tuning-Methoden für domänenspezifische Datensätze
  vergleichen
- Untersuchen, wie viele domänenspezifische Daten für effektives Fine-Tuning
  benötigt werden

**Notizen:**

- Datensatz ist sehr wichtig
- Sehr grosser Datensatz
- 10'000 zur seite für default TabPFN
- grösserer Chunk für fine tunign verwenden
- Auf dem fine tuned Modell die gleichen 10'000 verwenden
- mit zb random forest vergelichen auf grossem datensatz
